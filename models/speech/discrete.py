import torch.nn as nn
import torch
from hparams.hp import Hparams
from utils.tensormask import TensorMask
from modules.transformer.layers import TransformerLayerStack
from modules.linear.layers import RVQEmbedding, Embedding
from models.vocoder.vocoder import SoundStreamIO, HuBERTIO
from typing import Mapping, List, Optional, Union


class ARCTransformer(nn.Module):
    def __init__(self, hp: Hparams,
                 num_quantizers: int,
                 codebook_size: int,
                 embedding_dim: int) -> None:
        assert num_quantizers > 1
        super().__init__()
        self.hp = hp
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.pos_encoding = nn.Parameter(
            torch.randn(num_quantizers, embedding_dim)
        )
        self.transformer = TransformerLayerStack(hp,
                                                 input_dim=embedding_dim,
                                                 output_dim=codebook_size)
        self.embedding = nn.Embedding((num_quantizers - 1) * codebook_size,
                                      embedding_dim)

    def forward(self, x: TensorMask, x_label: TensorMask) -> TensorMask:
        """
        x: B, T, C
        x_label: B, T, n
        output: B, T, n, C
        """
        b, t, c = x.size()
        mask = x.mask
        x_label = x_label.value[..., :-1]  # Remove last code
        shift = torch.arange(self.num_quantizers - 1,
                             dtype=torch.long,
                             device=x_label.device)[None, None]
        x_label = x_label + shift * self.codebook_size
        embedding = self.embedding(x_label)  # B, T, n-1, C
        _input = torch.cat([x.value[:, :, None], embedding], 2)  # B, T, n, C
        _input = _input.reshape([b*t, self.num_quantizers, self.embedding_dim])
        _input = _input + self.pos_encoding[None]
        output = self.transformer(TensorMask(_input))  # BT, n, C
        output = output.value.reshape([b, t,
                                       self.num_quantizers,
                                       self.codebook_size])
        return TensorMask(output, mask).apply_mask()

    def step(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        x: [B, 1, C], [B], [B], ...
        output: B, 1, nc
        """
        o = x[0].squeeze(1)
        b, c = o.size()
        if len(x) > 1:
            if len(x) > 2:
                x_label = torch.stack(x[1:], -1)  # B, n-1
                shift = torch.arange(len(x) - 1,
                                     dtype=torch.long,
                                     device=o.device)[None]
                x_label = x_label + shift * self.codebook_size
            else:
                x_label = x[1][..., None]
            embedding = self.embedding(x_label)  # B, n-1, C
            _input = torch.cat([o[:, None], embedding], 1)  # B, n, C
        else:
            _input = o[:, None]
        _input = _input + self.pos_encoding[None, :len(x)]
        output = self.transformer(TensorMask(_input))  # BT, n, C
        return output.value[:, -1]


class DiscreteAR(nn.Module):
    def __init__(self, hp: Hparams,
                 hp_vq: Hparams,
                 input_dim: Optional[int] = None) -> None:
        super().__init__()
        hp.check_arg_in_hparams("transformer")
        self.input_dim = input_dim
        self.hp = hp
        self.hp_vq = hp_vq
        self.f0 = hp.get("f0", None)
        # Add SOS and EOS Embeddings: ..., SOS, EOS
        if hp_vq.num_quantizers > 1:
            self.single_vq = False
            hp.check_arg_in_hparams("arc_transformer")
            self.transformer = nn.Sequential(
                RVQEmbedding(hp_vq.num_quantizers,
                             hp_vq.codebook_size + 2,
                             hp_vq.dim),
                TransformerLayerStack(hp.transformer,
                                      input_dim=hp_vq.dim)
            )
            self.arc_transformer = ARCTransformer(hp.arc_transformer,
                                                  hp_vq.num_quantizers,
                                                  hp_vq.codebook_size,
                                                  hp.transformer.layer.dim)
        else:
            input_dim = hp_vq.dim
            if self.f0 is not None:
                input_dim += 1
            self.single_vq = True
            self.transformer = nn.Sequential(
                Embedding(hp_vq.codebook_size + 2,
                          hp_vq.dim),
                TransformerLayerStack(hp.transformer,
                                      input_dim=input_dim,
                                      output_dim=hp_vq.codebook_size)
            )
        if self.f0 is not None:
            self.f0_dense = nn.Linear(hp.transformer.layer.dim, 1)
        self.soundstream = None

    def set_soundstream(self,
                        soundstream: Union[SoundStreamIO, HuBERTIO]
                        ) -> None:
        self.soundstream = soundstream
        for param in self.soundstream.parameters():
            param.requires_grad = False

    @property
    def sample_ratio(self) -> float:
        return self.soundstream.sample_ratio

    def forward(self,
                x: TensorMask,
                c: Optional[TensorMask] = None,
                f0: Optional[TensorMask] = None,
                ) -> Mapping[str, TensorMask]:
        x = self.soundstream.encode_mel(x)
        shifted_x = x.push(
            self.initial_state(x.value.shape[0], x.value.device)
        ).pop().apply_mask()
        if self.f0 is not None:
            f0_dummy = torch.zeros((f0.value.size(0), 1),
                                   device=f0.value.device,
                                   dtype=f0.value.dtype)
            f0 = f0.push(f0_dummy).pop().apply_mask()
            shifted_x = self.transformer[0](shifted_x)
            shifted_x = shifted_x.cat(TensorMask(f0.value[..., None], f0.mask))
        else:
            shifted_x = self.transformer[0](shifted_x)
        transformed_x = self.transformer[1].run(shifted_x, c)
        if self.f0 is not None:
            f0 = TensorMask(
                self.f0_dense(transformed_x['layers'][-1].value),
                f0.mask)
        transformed_x = transformed_x['output']
        if not self.single_vq:
            logits = self.arc_transformer(transformed_x, x)
        else:
            logits = transformed_x
        output = {
            'logits': logits,
            'labels': x
        }
        if self.f0 is not None:
            output['f0'] = f0
        return output

    def step(self,
             x: torch.Tensor,
             c: Optional[TensorMask] = None,
             past_kv: Optional[List] = None,
             temperature: float = 1.0,
             return_attn: bool = False,
             return_distrbution: bool = False,
             **kwargs) -> Mapping:
        """
        x: B, 1, C
        """
        b = x.size(0)
        outputs = dict()
        if self.f0 is not None:
            f0 = x[..., -1:]
            x = x[..., 0]
        x = self.transformer[0](TensorMask(x.long()))
        if self.f0 is not None:
            x = x.cat(TensorMask(f0))
        z_given = self.transformer[1].run(x,
                                          memory=c,
                                          past_kv=past_kv,
                                          return_attn=return_attn,
                                          return_kv=True)
        if not self.single_vq:
            arc_inputs = [z_given["output"].value[:, -1:]]
            for i in range(self.hp_vq.num_quantizers):
                logits = self.arc_transformer.step(arc_inputs)
                probs = torch.softmax(logits / temperature, dim=-1)
                probs = probs.reshape([b, -1])
                sample = torch.multinomial(probs, 1)
                sample = sample.reshape([b, 1, -1])[..., -1, 0]  # B,
                arc_inputs.append(sample)
            outputs["output"] = torch.stack(arc_inputs[1:], -1).unsqueeze(1)
        else:
            logits = z_given["output"].value[:, -1]
            probs = torch.softmax(logits / temperature, dim=-1)
            probs = probs.reshape([b, -1])
            sample = torch.multinomial(probs, 1)
            outputs["output"] = sample
        outputs["kv"] = z_given["kv"]
        if return_attn:
            outputs["attn"] = z_given["self_attn"],
        if self.f0 is not None:
            f0 = self.f0_dense(z_given["layers"][-1].value[:, -1])
            outputs["output"] = torch.cat(
                [outputs["output"][..., None], f0[..., None]], -1)
        return outputs

    def decode(self, x: TensorMask,
               spkr: Optional[TensorMask] = None) -> TensorMask:
        args = [x]
        if self.f0 is not None:
            f0 = TensorMask(x.value[..., -1], x.mask)
            x = TensorMask(x.value[..., 0].long(), x.mask)
            args = [x]
        if spkr is not None:
            args.append(spkr)
        if self.f0 is not None:
            args.append(f0)
        return self.soundstream.decode(*args).apply_mask()

    def encode(self, x: TensorMask,
               temperature: float = 1.0) -> TensorMask:
        output = self.soundstream.encode_mel(x)
        return output.apply_mask()

    def initial_state(self, bsize: int, device=None):
        if self.single_vq:
            size = [bsize, 1]
        else:
            size = [bsize, 1, self.hp_vq.num_quantizers]
        return torch.full(size,
                          fill_value=self.hp_vq.codebook_size,
                          dtype=torch.long,
                          device=device)

    def likelihood(self, x: TensorMask,
                   f0: Optional[TensorMask] = None,
                   **kwargs) -> TensorMask:
        out = self.forward(x, f0=f0)
        logits, labels = out['logits'], out['labels']
        log_probs = torch.log_softmax(logits.value, dim=-1)
        b, t, c = logits.value.size()
        index = labels.value.reshape([-1])
        index = index + torch.arange(index.size(0),
                                     device=index.device) * c
        log_probs = log_probs.reshape([-1])[index]
        log_probs = log_probs.reshape([b, t])
        log_probs = TensorMask.use_mask(log_probs, logits.mask)
        return log_probs.sum(-1) / logits.length
