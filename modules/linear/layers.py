import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.tensormask import TensorMask
from utils.helpers import repeat_batch
from utils.attr import AttrDict
from typing import Optional, Union, Tuple
from hparams.hp import Hparams
from modules.norm import get_norm_fn
from modules.activations import get_activation
from vector_quantize_pytorch import VectorQuantize


class VectorQuantizeParameterize(nn.Module):
    def __init__(self,
                 in_dim: int,
                 codebook_size: int):
        super().__init__()
        self.vq = VectorQuantize(
            dim=in_dim,
            codebook_size=codebook_size,
            sample_codebook_temp=0.0,
            learnable_codebook=True,
            commitment_weight=0.2,
            ema_update=False
        )

    def forward(self, x: TensorMask) -> AttrDict:
        mask = x.mask
        q, ind, loss = self.vq(x.value, mask=mask)
        loss = loss * mask.float().sum()
        return AttrDict(
            indicies=TensorMask(ind, mask).apply_mask(),
            output=TensorMask(q, mask).apply_mask(),
            loss=loss
        )

    def get_output(self, ind: torch.Tensor) -> torch.Tensor:
        return self.vq.codebook[ind]


class GumbelSoftMaxParameterize(nn.Module):
    def __init__(self,
                 in_dim: int,
                 num_codebooks: int,
                 codebook_dim: int,
                 temperature: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.in_linear = nn.Linear(in_dim, num_codebooks, bias=False)
        self.encode_linear = nn.Linear(num_codebooks, codebook_dim, bias=False)
        self.temperature = temperature

    def gumbel_softmax_sample(self, logits, temperature, eps=1e-20):
        U = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(U + eps) + eps)
        y = logits + gumbel
        return F.softmax(y / temperature, dim=-1)

    def forward(self, x: TensorMask,
                temperature: Optional[float] = None
                ) -> AttrDict:
        mask = x.mask
        logits = self.in_linear(x.value) / self.in_dim ** 0.5
        if temperature is None:
            temperature = self.temperature
        y = self.gumbel_softmax_sample(logits, temperature)
        gumbel_prob = y
        b, t, c = y.size()
        y = y.reshape([b*t, c])
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).scatter(1, ind.view(-1, 1), 1)
        y_hard = (y_hard - y).detach() + y
        y_hard = y_hard.reshape([b, t, c])
        output = self.encode_linear(y_hard)
        return AttrDict(
            logits=TensorMask(logits, mask).apply_mask(-1000),
            output=TensorMask(output, mask).apply_mask(),
            gumbel_prob=TensorMask(gumbel_prob, mask).apply_mask()
        )


class GaussianParameterize(nn.Module):
    def __init__(self, in_dim: int, dim: int,
                 bias: bool = True,
                 std: Optional[float] = None,
                 std_range: Optional[Tuple[float, float]] = None,
                 truncated_norm: Optional[Tuple[float, float]] = None,
                 total_std: Optional[float] = None,
                 use_tanh: bool = False,
                 use_relu: bool = False,
                 normalization: bool = False,
                 mean: Optional[float] = None):
        super().__init__()
        self._mean = mean
        self.dim = dim
        if mean is None:
            self.mean = nn.Linear(in_dim, dim, bias=bias)
        self.std = std
        self.truncated_norm = truncated_norm
        if std is None:
            self.logstd = nn.Linear(in_dim, dim, bias=bias)
        self.std_range = None
        if std_range is not None:
            assert std is None
            assert len(std_range) == 2
            self.std_range = std_range
        self.total_std = total_std
        if self.total_std is not None:
            assert std is None
            assert std_range is None
        self.use_tanh = use_tanh
        self.use_relu = use_relu
        self.normalization = normalization

    def forward(self, x: TensorMask,
                temperature: float = 1.0,
                truncated_norm: Optional[Tuple[float, float]] = None
                ) -> AttrDict:
        if self._mean is None:
            mean = self.mean(x.value)
        else:
            mean = torch.full([x.value.size(0),
                               x.value.size(1),
                               self.dim],
                              self._mean,
                              device=x.device)
        if self.normalization:
            mean = torch.nn.functional.normalize(mean, p=2.0, dim=-1)
        if self.use_relu:
            mean = F.relu(mean)
        if self.use_tanh:
            mean = torch.tanh(mean) * 0.5
        if self.std is None:
            logstd = self.logstd(x.value)
            if self.std_range is not None:
                _max, _min = self.std_range
                std = torch.sigmoid(logstd) * (_max - _min) + _min
                logstd = torch.log(std)
        else:
            logstd = torch.log(torch.full(mean.size(), self.std,
                                          device=x.device))
        noise = torch.randn_like(mean)
        if self.truncated_norm is not None:
            nn.init.trunc_normal_(noise,
                                  a=self.truncated_norm[0],
                                  b=self.truncated_norm[1])
        if truncated_norm is not None:
            nn.init.trunc_normal_(noise,
                                  a=truncated_norm[0], b=truncated_norm[1])
        std = torch.exp(logstd.float())
        if self.total_std is not None:
            std = std / std.sum(-1, keepdim=True)
            std = std * self.total_std * std.size(-1)
            logstd = torch.log(std)
        noise = noise * std
        sample = mean + noise * temperature
        output = AttrDict(
            mean=TensorMask(mean, x.mask),
            logstd=TensorMask(logstd, x.mask),
            sample=TensorMask(sample, x.mask)
        )
        return output

    def sample(self, n: int,
               mean: TensorMask, logstd: TensorMask,
               temperature: float = 1.0) -> AttrDict:
        mean, logstd = repeat_batch(mean, n), repeat_batch(logstd, n)
        noise = torch.randn_like(mean.value) * torch.exp(logstd.value.float())
        sample = mean.value + noise * temperature
        output = AttrDict(
            mean=mean,
            logstd=logstd,
            sample=TensorMask(sample, mean.mask)
        )
        return output


class Embedding(nn.Embedding):
    def forward(self, x: TensorMask):
        return TensorMask(super().forward(x.value), x.mask).apply_mask()

    def custom_weight_init(self, init_std: float):
        self._fill_padding_idx_with_zero()
        std = 1.0
        self.weight.data.uniform_(-std, std)


class RVQEmbedding(nn.Module):
    def __init__(self,
                 num_quantizers: int,
                 codebook_size: int,
                 dim: int) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, dim)
            for _ in range(num_quantizers)
        ])

    def forward(self, x: TensorMask):
        '''
        x: B, T, n
        out: B, T, C
        '''
        output = 0
        mask = x.mask
        for i in range(self.num_quantizers):
            output += self.embeddings[i](x.value[..., i])
        return TensorMask(output, mask).apply_mask()


class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 bias: bool = True,
                 activation=nn.Identity()) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation

    def forward(self, x: TensorMask):
        return TensorMask(self.activation(self.linear(x.value)), x.mask)


class LinearBlock(nn.Module):
    def __init__(self,
                 hp: Hparams):
        super().__init__()
        bias = hp.get("bias", True)
        hp.check_arg_in_hparams("hidden_dim",
                                "activation",
                                "norm")
        dropout = hp.get("dropout", 0.0)
        self.linear1 = nn.Linear(hp.hidden_dim, hp.hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hp.hidden_dim,
                                 hp.hidden_dim,
                                 bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = get_norm_fn(hp.hidden_dim, hp.norm)
        self.norm2 = get_norm_fn(hp.hidden_dim, hp.norm)
        self.activation = get_activation(hp.activation)

    def forward(self, x: TensorMask) -> TensorMask:
        r = self.linear1(
            self.activation(
              self.norm1(x.value)
            )
        )
        r = self.linear2(
            self.activation(
              self.norm2(r)
            )
        )
        return TensorMask(
            x.value + r,
            x.mask,
        ).apply_mask()


class LinearLayerStack(nn.Module):
    def __init__(self, hp: Hparams,
                 input_dim: Optional[int] = None,
                 output_dim: Optional[int] = None) -> None:
        super().__init__()
        self.hp = hp
        hp.check_arg_in_hparams("num_layers",
                                "layer")
        self.layers = nn.ModuleList([
            LinearBlock(hp.layer) for _ in range(hp.num_layers)
        ])
        self.linear = None
        if input_dim is not None:
            self.linear = nn.Linear(input_dim, hp.layer.hidden_dim)
        self.out_linear = None
        if output_dim is not None:
            self.out_linear = nn.Linear(hp.layer.hidden_dim, output_dim)

    def forward(self, x: TensorMask) -> TensorMask:
        """x: BTC TensorMask."""
        if self.linear is not None:
            x = TensorMask(self.linear(x.value), x.mask).apply_mask()
        for layer in self.layers:
            x = layer(x)
        if self.out_linear is not None:
            x = TensorMask(self.out_linear(x.value), x.mask).apply_mask()
        return x


class TimeAggregation(nn.Module):
    def forward(self, x: TensorMask) -> torch.Tensor:
        return x.flatten().apply_mask().value.sum(1) / x.length[..., None]


class FiLM(nn.Module):
    def __init__(self, dim: int, bias: bool = True,
                 time_first: bool = True,
                 in_dim: int = None):
        super().__init__()
        if in_dim is None:
            in_dim = dim
        if time_first:
            self.linear = nn.Linear(in_dim, dim * 2, bias=bias)
        else:
            self.linear = nn.Conv1d(in_dim, dim * 2, 1, bias=bias)
        self.time_first = time_first

    def forward(self,
                x: Union[torch.Tensor, TensorMask],
                c: Union[torch.Tensor, TensorMask]):
        y = x
        if isinstance(x, TensorMask):
            y = y.value
        if isinstance(c, TensorMask):
            c = c.value
        axis = -1 if self.time_first else 1
        weight, bias = self.linear(c).chunk(2, axis)
        y = weight * y + bias
        if isinstance(x, TensorMask):
            axis = 1 if self.time_first else 2
            return TensorMask(y, x.mask, axis=axis)
        return y
