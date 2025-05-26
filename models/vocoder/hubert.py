import torch.nn as nn
import torch
from hparams.hp import Hparams
from modules.conv.layers import CNNStack, ResNet
from modules.diffusion.unet import ConditionalBottleNeckUNet
from modules.diffusion.ddpm import GaussianDiffusion1D
from modules.linear.layers import Embedding, TimeAggregation
from typing import Mapping, Optional
from utils.tensormask import TensorMask
from utils.helpers import interpolate, pad_1d


class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                duration: torch.Tensor) -> TensorMask:
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])
        max_len = max([x.size(0) for x in output])
        output = [pad_1d(x, 1, max_len) for x in output]
        output = torch.stack(output, 0)
        mel_len = torch.LongTensor(mel_len).to(x.device)
        return TensorMask.fromlength(output, mel_len)

    def expand(self, batch: torch.Tensor,
               predicted: torch.Tensor) -> torch.Tensor:
        out = list()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)
        return out


class HuBERT(nn.Module):
    def __init__(self, hp: Hparams,
                 input_dim: Optional[int] = None,
                 mel_sample_rate: Optional[int] = None) -> None:
        super().__init__()
        hp.check_arg_in_hparams("hubert",
                                "embed_encoder",
                                "decoder")
        self.hp = hp
        self.input_dim = input_dim
        self.embedding = Embedding(hp.hubert.vocab_size,
                                   hp.embedding_dim)
        self.deduplicate = hp.hubert.deduplicate
        self.spkr_encoder = None
        _embed_dim = hp.embedding_dim
        if hp.has('spkr'):
            self.spkr_encoder = nn.Sequential(
                CNNStack(hp.spkr,
                         input_dim=input_dim,
                         output_dim=hp.spkr.embedding_dim),
                TimeAggregation()
            )
            _embed_dim = hp.embedding_dim + hp.spkr.embedding_dim
        self.f0 = None
        if hp.has('f0'):
            self.f0 = True
            _embed_dim += 1
        self.embed_encoder = ResNet(hp.embed_encoder,
                                    input_dim=_embed_dim,
                                    output_dim=hp.embedding_dim)
        if self.deduplicate:
            hp.check_arg_in_hparams('duration_predictor')
            self.dp = ResNet(hp.duration_predictor,
                             input_dim=_embed_dim,
                             output_dim=1)
            self.upsampler = LengthRegulator()
        model = ConditionalBottleNeckUNet(hp.embedding_dim,
                                          input_dim,
                                          hp.decoder.cond_unet)
        self.decoder = GaussianDiffusion1D(model, hp.decoder.diffusion)
        self.diff_scaling = hp.decoder.diffusion.get("input_scale", 1.0)
        self.interpolate_ratio = hp.get('interpolate_ratio', None)
        self.mel_sample_rate = mel_sample_rate

    def forward(self,
                x: TensorMask,
                x_mel: TensorMask,
                spkr: Optional[TensorMask] = None,
                dedup_x: Optional[TensorMask] = None,
                f0: Optional[TensorMask] = None
                ) -> Mapping[str, TensorMask]:
        x = self.embedding(x)
        if self.f0 is not None:
            x = x.cat(TensorMask(f0.value[..., None], f0.mask))
        if self.spkr_encoder is not None:
            spkr = self.spkr_encoder(spkr)
            _spkr = spkr[:, None].expand(-1, x.value.size(1), -1)
            x = x.cat(_spkr)
        x = self.embed_encoder(x)
        if self.interpolate_ratio is not None:
            x = interpolate(x, self.interpolate_ratio)
        diffusion_loss = self.decoder(x_mel / self.diff_scaling, x)
        output = {
            'diffusion_loss': diffusion_loss,
            'condition': x
        }
        if self.deduplicate:
            dedup_x = self.embedding(dedup_x)
            if self.spkr_encoder is not None:
                _spkr = spkr[:, None].expand(-1, dedup_x.value.size(1), -1)
                dedup_x = dedup_x.cat(_spkr)
            dp = self.dp(dedup_x)
            output['duration_prediction'] = dp
        return output

    def decode(self, x: TensorMask) -> TensorMask:
        if self.interpolate_ratio is not None:
            intr = float(self.interpolate_ratio)
        else:
            intr = 1.0
        noise_shape = [x.value.size(0),
                       int(x.value.size(1) / intr
                           * self.sample_ratio),
                       self.input_dim]
        noise = torch.randn(noise_shape, device=x.device)
        noise = TensorMask.fromlength(
            noise,
            TensorMask.resize_length(x.length, self.sample_ratio)
        ).apply_mask()
        return self.decoder.sample(noise, x.apply_mask()) * self.diff_scaling

    def encode(self,
               x: TensorMask,
               spkr: Optional[TensorMask] = None,
               f0: Optional[TensorMask] = None
               ) -> TensorMask:
        if self.spkr_encoder is not None:
            spkr = self.spkr_encoder(spkr)
        if self.deduplicate:
            dedup_x = self.embedding(x)
            if self.f0 is not None:
                x = x.cat(TensorMask(f0.value[..., None], f0.mask))
            if self.spkr_encoder is not None:
                _spkr = spkr[:, None].expand(-1, x.value.size(1), -1)
                dedup_x = dedup_x.cat(_spkr)
            dp = self.dp(dedup_x)
            duration = torch.exp(dp.value) - 1
            duration = torch.clamp(duration, min=1.0)
            duration = torch.ceil(duration)
            duration = TensorMask.use_mask(duration, dp.mask).long()
            x = self.upsampler(dedup_x.value, duration.squeeze(-1))
        else:
            x = self.embedding(x)
            if self.f0 is not None:
                x = x.cat(TensorMask(f0.value[:, :x.value.size(1), None],
                                     x.mask))
            if self.spkr_encoder is not None:
                _spkr = spkr[:, None].expand(-1, x.value.size(1), -1)
                x = x.cat(_spkr)
        x = self.embed_encoder(x)
        if self.interpolate_ratio is not None:
            x = interpolate(x, self.interpolate_ratio)
        return x

    @property
    def sample_ratio(self) -> float:
        return float(self.mel_sample_rate) / float(self.hp.hubert.sample_rate)
