import torch.nn as nn
import torch
import math
from hparams.hp import Hparams
from utils.tensormask import TensorMask
from modules.transformer.layers import TransformerLayerStack
from modules.flow.layers import CouplingStack
from modules.flow.utils import TensorLogdet
from modules.conv.layers import ResNet, BottleNeckResNet, CNNStack
from modules.linear.layers import (GaussianParameterize,
                                   TimeAggregation, Embedding, Linear)
from modules.diffusion.ddpm import GaussianDiffusion1D
from modules.diffusion.unet import ConditionalUNet, ConditionalBottleNeckUNet
from typing import Mapping, List, Optional, Tuple
from training_lib.losses import masked_ce_loss


class LVTR(nn.Module):
    def __init__(self, hp: Hparams,
                 input_dim: Optional[int] = None,
                 memory_dim: Optional[int] = None) -> None:
        super().__init__()
        hp.check_arg_in_hparams("encoder",
                                "decoder",
                                "transformer",
                                "latent_dim")
        self.input_dim = input_dim
        self.hp = hp
        encoder_identifier = hp.encoder.get("identifier", "ResNet")
        if encoder_identifier == 'BottleNeckResNet':
            encoder_model = BottleNeckResNet
        elif encoder_identifier == 'ResNet':
            encoder_model = ResNet
        elif encoder_identifier == 'CNNStack':
            encoder_model = CNNStack
        else:
            raise ValueError(f"{encoder_identifier} not recoginized.")

        connection_dim = hp.latent_dim
        self.encoder = nn.Sequential(
            encoder_model(hp.encoder,
                          input_dim=input_dim,
                          output_dim=connection_dim),
        )
        self.encoder.append(
            GaussianParameterize(
                connection_dim,
                hp.latent_dim,
                std=hp.encoder.get('fix_std', None),
                std_range=hp.encoder.get('std_range', None),
                truncated_norm=hp.encoder.get('truncated_norm', None),
                total_std=hp.encoder.get('total_std', None),
                use_tanh=False,
                normalization=hp.encoder.get('normalization', False)
            )
        )

        self.tokens = hp.get("tokens", None)
        if self.tokens is not None:
            self.tokens.check_arg_in_hparams("embedding_dim",
                                             "vocab_size")
            self.token_embedding_dim = self.tokens.embedding_dim
            self.token_embedding = Embedding(self.tokens.vocab_size,
                                             self.tokens.embedding_dim)
            self.token_predictor = Linear(hp.transformer.layer.dim,
                                          self.tokens.vocab_size)
            self.token_fuser = Linear(
                hp.latent_dim,
                self.tokens.embedding_dim,
                activation=nn.ReLU()
            )
            self.token_spliter = Linear(hp.transformer.layer.dim,
                                        hp.transformer.layer.dim,
                                        activation=nn.ReLU())
            self.q_spliter = Linear(hp.transformer.layer.dim,
                                    hp.transformer.layer.dim,
                                    activation=nn.ReLU())
        else:
            self.q_spliter = nn.Identity()
        diff_cond_dim = hp.latent_dim
        if self.tokens is not None:
            diff_cond_dim = self.tokens.embedding_dim
        if hp.has("utterance_encoder"):
            diff_cond_dim += hp.utterance_encoder.embedding_dim
        decoder_identifier = hp.decoder.diffusion.get("identifier",
                                                      "ConditionalUNet")
        if decoder_identifier == 'ConditionalBottleNeckUNet':
            decoder_model = ConditionalBottleNeckUNet
        elif decoder_identifier == 'ConditionalUNet':
            decoder_model = ConditionalUNet
        else:
            raise ValueError(f"{encoder_identifier} not recoginized.")
        hp.decoder.check_arg_in_hparams("cond_unet")
        model = decoder_model(diff_cond_dim, input_dim,
                              hp.decoder.cond_unet)
        self.decoder = GaussianDiffusion1D(model, hp.decoder.diffusion)
        self.diff_scaling = hp.decoder.diffusion.get("input_scale", 1.0)

        self.transformer_flow = None
        if hp.transformer.has("flow"):
            condition_dim = hp.transformer.layer.dim
            if not hp.transformer.flow.get("conditional", False):
                condition_dim = None
            self.transformer_flow = CouplingStack(
                hp.latent_dim,
                hp.transformer.flow,
                condition_dim=condition_dim
            )
        tr_input_dim = hp.latent_dim
        if self.tokens is not None:
            tr_input_dim = self.tokens.embedding_dim
        self.transformer = nn.Sequential(
            TransformerLayerStack(hp.transformer,
                                  input_dim=tr_input_dim,
                                  memory_dim=memory_dim),
        )
        condition_dim = hp.transformer.layer.dim
        self.transformer.append(
            GaussianParameterize(condition_dim,
                                 hp.latent_dim,
                                 std=hp.transformer.get('fix_std', None),
                                 std_range=hp.transformer.get('std_range',
                                                              None),
                                 use_tanh=False,
                                 mean=hp.transformer.get('fix_mean', None))
        )
        self.utterance_encoder = None
        if hp.has("utterance_encoder"):
            self.utterance_encoder = nn.Sequential(
                CNNStack(
                    hp.utterance_encoder,
                    input_dim=input_dim,
                    output_dim=hp.utterance_encoder.embedding_dim
                ),
                TimeAggregation()
            )
        self.use_tokens = self.tokens is not None

    @property
    def sample_ratio(self) -> float:
        return self.encoder[0].sample_ratio

    def forward(self,
                x: TensorMask,
                c: Optional[TensorMask] = None,
                spkr: Optional[torch.Tensor] = None,
                utterance: Optional[TensorMask] = None,
                diff_input: Optional[TensorMask] = None
                ) -> Mapping[str, TensorMask]:
        # use x to overload: [tokens, x]
        if self.use_tokens:
            tokens_id, x = self.split_inputs(x)
            tokens_id = tokens_id.long().squeeze(-1)
            tokens = self.token_embedding(tokens_id)
        encode_x = x
        q_z = self.encoder(encode_x)
        sample_q = q_z.sample
        log_q = -q_z.logstd.value - 0.5 - 0.5 * math.log(2 * math.pi)
        log_q = TensorMask(log_q, q_z.logstd.mask)
        sample_q = sample_q.apply_mask()
        init_state = self.initial_state(x.value.shape[0], x.value.device)
        n_shift = 1
        init_state = init_state.expand(-1, n_shift, -1)
        shifted_sample_q = sample_q
        if self.use_tokens:
            shifted_sample_q = self.fuse_inputs(shifted_sample_q, tokens)
        shifted_sample_q = shifted_sample_q.push(init_state)
        shifted_sample_q = shifted_sample_q.pop(n_shift).apply_mask()
        transformer_input = shifted_sample_q
        transformer_latent = self.transformer[0](transformer_input, c)
        q_split_latent = self.q_spliter(transformer_latent)
        z_given = self.transformer[1](q_split_latent)
        if self.transformer_flow is None:
            log_p = -z_given.logstd.value - 0.5 * math.log(2 * math.pi)
            log_p += -0.5 * (torch.exp(-2 * z_given.logstd.value) *
                             (sample_q.value - z_given.mean.value) ** 2)
        else:
            target_logstd = z_given.logstd
            target_mean = z_given.mean
            condition = q_split_latent
            input_sample_q = sample_q
            p_z = self.transformer_flow(TensorLogdet(input_sample_q, 0.0),
                                        c=condition)
            sample_p, logdet_p = p_z.tensor, p_z.logdet
            log_p = logdet_p.sum(-1) / self.hp.latent_dim
            log_p = log_p[..., None]
            log_p = log_p - target_logstd.value
            log_p = log_p - 0.5 * math.log(2 * math.pi)
            log_p += -0.5 * (torch.exp(-2 * target_logstd.value) *
                             (sample_p.value - target_mean.value) ** 2)
        log_p = TensorMask(log_p, z_given.sample.mask)
        ce_loss = None
        if self.use_tokens:
            pred_tokens = self.token_predictor(
                self.token_spliter(transformer_latent))
            ce_loss = masked_ce_loss(pred_tokens, tokens_id)
        if diff_input is None:
            diffusion_input = sample_q
        else:
            diffusion_input = self.encoder(diff_input).sample
        if self.use_tokens:
            diffusion_input = self.fuse_inputs(diffusion_input, tokens)
        u_c = None
        if self.utterance_encoder is not None:
            u_c = self.utterance_encoder(utterance)
            diffusion_input = diffusion_input.cat(
                u_c[:, None].expand(-1, diffusion_input.value.size(1), -1))
        xi = x if diff_input is None else diff_input
        rec_x = self.decoder(xi / self.diff_scaling, diffusion_input)
        output = {
            'log_p': log_p.apply_mask(),
            'log_q': log_q.apply_mask(),
            'decoder_output': rec_x,
            'sample_q': sample_q,
            'transformer_latent': transformer_latent,
            'logstd': z_given.logstd.mean(),
            'mean': z_given.mean.mean(),
            'q_logstd': q_z.logstd.mean(),
            'q_mean': q_z.mean.mean(),
            'q_z': q_z,
            'u_c': u_c,
            'q_mean_abs': q_z.mean.abs().mean(),
            'ce_loss': ce_loss
        }
        return output

    def step(self,
             x: torch.Tensor,
             c: Optional[TensorMask] = None,
             spkr: Optional[torch.Tensor] = None,
             past_kv: Optional[List] = None,
             temperature: float = 1.0,
             token_temperature: float = 1.0,
             truncated_norm: Optional[Tuple[float, float]] = None,
             return_attn: bool = False,
             return_distrbution: bool = False,
             push_init_state: bool = False,
             **kwargs) -> Mapping:
        """
        x: B, 1, C
        """
        x = TensorMask(x)
        if self.use_tokens:
            tokens_id, x = self.split_inputs(x)
            tokens_id = tokens_id.long().squeeze(-1)
            tokens = self.token_embedding(tokens_id)
            x = self.fuse_inputs(x, tokens)
        if push_init_state:
            x = x.push(
                self.initial_state(x.value.shape[0], x.value.device)
            ).apply_mask()
        outputs = dict()
        z_given = self.transformer[0].run(x,
                                          memory=c,
                                          past_kv=past_kv,
                                          return_attn=return_attn,
                                          return_kv=True)
        outputs['transformer_latent'] = z_given["output"]
        if return_distrbution:
            outputs["z_given"] = z_given
        outputs["kv"] = z_given["kv"]
        if return_attn:
            outputs["self_attn"] = z_given["self_attn"]
            if "cross_attn" in z_given:
                outputs["cross_attn"] = z_given["cross_attn"]

        q_split = self.q_spliter(outputs['transformer_latent'])
        sample_z = self.transformer[1](
            q_split,
            temperature=temperature,
            truncated_norm=truncated_norm).sample
        if self.transformer_flow is not None:
            sample_z = self.transformer_flow.reverse(
                sample_z,
                c=q_split)
        outputs["output"] = sample_z.value
        if self.use_tokens:
            logits = self.token_predictor(
                self.token_spliter(z_given["output"])).value
            b, t, c = logits.size()
            probs = torch.softmax(logits / token_temperature, dim=-1)
            probs = probs.reshape([b*t, c])
            sample = torch.multinomial(probs, 1)
            sample = sample.reshape([b, t, 1]).float()
            outputs["output"] = torch.cat([sample, outputs["output"]], -1)
        return outputs

    def decode(self, x: TensorMask,
               c: Optional[TensorMask] = None,
               u_c: Optional[torch.Tensor] = None) -> TensorMask:
        noise_shape = [x.value.size(0),
                       int(x.value.size(1) * (1.0 / self.sample_ratio)),
                       self.input_dim]
        noise = torch.randn(noise_shape, device=x.device)
        noise = TensorMask.fromlength(
            noise,
            TensorMask.resize_length(x.length, 1.0 / self.sample_ratio)
        ).apply_mask()
        if self.use_tokens:
            tokens_id, x = x.split(1)
            tokens_id = tokens_id.long().squeeze(-1)
            tokens = self.token_embedding(tokens_id)
            x = self.fuse_inputs(x, tokens)
        if u_c is not None:
            x = x.cat(u_c[:, None].expand(-1, x.value.size(1), -1))
        return self.decoder.sample(noise, x.apply_mask()) * self.diff_scaling

    def encode(self, x: TensorMask,
               temperature: float = 1.0,
               beta: Optional[torch.Tensor] = None,
               utterance: Optional[TensorMask] = None) -> TensorMask:
        if self.use_tokens:
            tokens_id, x = self.split_inputs(x)
        if beta is not None:
            x = self.encode_beta(beta, x)
        output = self.encoder[0](x)
        output = self.encoder[1](output, temperature).sample
        if self.use_tokens:
            return tokens_id.cat(output.apply_mask())
        return output.apply_mask()

    def encode_utterance(self, utterance: TensorMask) -> TensorMask:
        if self.use_tokens:
            tokens_id, utterance = self.split_inputs(utterance)
        u_c = self.utterance_encoder(utterance)
        return u_c

    def initial_state(self, bsize: int, device=None, nfeat=None):
        if nfeat is None:
            nfeat = self.hp.latent_dim
            if self.tokens is not None:
                nfeat = self.token_embedding_dim
        scale = 1.0
        return torch.rand(bsize, 1, nfeat,
                          device=device) * scale * 2 - scale

    def likelihood(self, x: TensorMask,
                   temperature: float = 0.0,
                   gamma: Optional[float] = 1.0,
                   **kwargs) -> TensorMask:
        if self.use_tokens:
            tokens_id, x = self.split_inputs(x)
            tokens_id = tokens_id.long().squeeze(-1)
            tokens = self.token_embedding(tokens_id)
        q = self.encoder[0](x)
        q = self.encoder[1](q, temperature).sample
        shift_q = q
        if self.use_tokens:
            shift_q = self.fuse_inputs(shift_q, tokens)
        shift_q = shift_q.push(
            self.initial_state(x.value.shape[0], x.value.device)
        ).pop().apply_mask()
        transformer_latent = self.transformer[0](shift_q)
        q_split = self.q_spliter(transformer_latent)
        z_given = self.transformer[1](q_split)
        if self.transformer_flow is not None:
            inverse_g_q = self.transformer_flow(
                TensorLogdet(q, 0.0),
                c=q_split
            )
            sample_p, logdet_p = inverse_g_q.tensor, inverse_g_q.logdet
            log_p = logdet_p.sum(-1) / self.hp.latent_dim
            log_p = log_p[..., None]
            log_p = log_p - z_given.logstd.value
            log_p = log_p - 0.5 * math.log(2 * math.pi)
            log_p += -0.5 * (torch.exp(-2 * z_given.logstd.value) *
                             (sample_p.value - z_given.mean.value) ** 2)
            log_p = TensorMask(log_p, sample_p.mask)
        else:
            log_p = -z_given.logstd.value - 0.5 * math.log(2 * math.pi)
            log_p += -0.5 * (torch.exp(-2 * z_given.logstd.value) *
                             (q.value - z_given.mean.value) ** 2)
            log_p = TensorMask(log_p, z_given.mean.mask)
        ret = log_p.apply_mask().value.mean(-1).sum(1) / log_p.length
        if self.use_tokens:
            logits = self.token_predictor(
                self.token_spliter(transformer_latent))
            labels = tokens_id
            log_probs = torch.log_softmax(logits.value, dim=-1)
            b, t, c = logits.value.size()
            index = labels.value.reshape([-1])
            index = index + torch.arange(index.size(0),
                                         device=index.device) * c
            log_probs = log_probs.reshape([-1])[index]
            log_probs = log_probs.reshape([b, t])
            log_probs = TensorMask.use_mask(log_probs, logits.mask)
            ret = log_probs.sum(-1) / logits.length  # * gamma
        return ret

    def fuse_inputs(self, x: TensorMask,
                    tokens: TensorMask) -> TensorMask:
        return tokens + self.token_fuser(x)

    def split_inputs(self, x: TensorMask):
        return x.split(1)
