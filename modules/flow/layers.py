import torch
import torch.nn as nn
from utils.tensormask import TensorMask
from utils.helpers import get_padding
from typing import Optional
from modules.activations import get_activation
from modules.norm import get_norm_fn
from hparams.hp import Hparams
from .spline import RationalQuadraticSplineCoupling
from .utils import TensorLogdet
from modules.conv.layers import Conv1d
from modules.linear.layers import FiLM


class LinearCoupling(nn.Module):
    def __init__(self,
                 dim: int,
                 flip: bool,
                 hp: Hparams,
                 condition_dim: Optional[int] = None):
        super().__init__()
        hp.check_arg_in_hparams("hidden_dim",
                                "activation",
                                "mean_only",
                                "norm")
        self.mean_only = hp.mean_only
        self.condition_dim = condition_dim
        if self.condition_dim is not None:
            self.film = FiLM(hp.hidden_dim, in_dim=condition_dim)
        self.linear1 = nn.Linear(dim // 2,
                                 hp.hidden_dim,
                                 bias=hp.get("bias", True))
        self.linear2 = nn.Linear(hp.hidden_dim,
                                 dim // 2 if hp.mean_only else dim,
                                 bias=hp.get("bias", True))
        self.norm = get_norm_fn(hp.hidden_dim, hp.norm)
        self.activation = get_activation(hp.activation)
        self.flip = flip
        self.scale_range = hp.get("scale_range", None)
        self.detach_coupling = hp.get('detach_coupling', False)

    def forward(self, x: TensorLogdet,
                c: Optional[TensorMask] = None) -> TensorLogdet:
        x0, x1 = x.tensor.value.chunk(2, -1)
        if self.flip:
            x0, x1 = x1, x0
        _input = x0
        if self.detach_coupling:
            _input = _input.detach()
        stats = self.norm(self.linear1(_input))
        if c is not None and self.condition_dim is not None:
            stats = self.film(stats, c)
        stats = self.linear2(
            self.activation(
              stats
            )
        )
        if self.mean_only:
            m, logs = stats, torch.zeros_like(stats)
        else:
            m, logs = stats.chunk(2, -1)
            if self.scale_range is not None:
                _max, _min = self.scale_range
                logs = torch.sigmoid(logs) * (_max - _min) + _min
                logs = torch.log(logs)
        x1 = m + x1 * torch.exp(logs)
        ret = torch.cat([x0, x1], -1)
        logs = TensorMask.use_mask(logs, x.tensor.mask)
        logdet = x.logdet + logs
        return TensorLogdet(
            TensorMask(ret, x.tensor.mask, axis=x.tensor.axis),
            logdet
        )

    def reverse(self, x: TensorMask,
                c: Optional[TensorMask] = None) -> TensorMask:
        x0, x1 = x.value.chunk(2, -1)
        _input = x0
        stats = self.norm(self.linear1(_input))
        if c is not None and self.condition_dim is not None:
            stats = self.film(stats, c)
        stats = self.linear2(
            self.activation(
              stats
            )
        )
        if self.mean_only:
            m, logs = stats, torch.zeros_like(stats)
        else:
            m, logs = stats.chunk(2, -1)
            if self.scale_range is not None:
                _max, _min = self.scale_range
                logs = torch.sigmoid(logs) * (_max - _min) + _min
                logs = torch.log(logs)
        x1 = (x1 - m) * torch.exp(-logs)
        if self.flip:
            x0, x1 = x1, x0
        ret = torch.cat([x0, x1], -1)
        return TensorMask(ret, x.mask, axis=x.axis)


class ConvCoupling(nn.Module):
    def __init__(self,
                 dim: int,
                 flip: bool,
                 hp: Hparams,
                 condition_dim: Optional[int] = None):
        super().__init__()
        hp.check_arg_in_hparams("hidden_dim",
                                "activation",
                                "mean_only",
                                "norm",
                                "kernel_size")
        self.mean_only = hp.mean_only
        self.condition_dim = condition_dim
        _condition_dim = 0
        if self.condition_dim is not None:
            _condition_dim = self.condition_dim
        causal_padding = hp.get("causal_padding", False)
        future_padding = hp.get("future_padding", False)
        self.conv1 = Conv1d(dim // 2 + _condition_dim,
                            hp.hidden_dim,
                            kernel_size=hp.kernel_size,
                            padding=get_padding(hp.kernel_size,
                                                causal=causal_padding,
                                                future=future_padding),
                            bias=hp.get("bias", False))
        self.conv2 = Conv1d(hp.hidden_dim,
                            dim // 2 if hp.mean_only else dim,
                            kernel_size=1,
                            padding=0,
                            bias=hp.get("bias", True))
        self.norm = get_norm_fn(hp.hidden_dim, hp.norm)
        self.activation = get_activation(hp.activation)
        self.flip = flip
        self.scale_range = hp.get("scale_range", None)
        self.detach_coupling = hp.get('detach_coupling', False)

    def forward(self, x: TensorLogdet,
                c: Optional[TensorMask] = None) -> TensorLogdet:
        x0, x1 = x.tensor.value.chunk(2, -2)
        if self.flip:
            x0, x1 = x1, x0
        _input = x0
        if self.detach_coupling:
            _input = _input.detach()
        if c is not None and self.condition_dim is not None:
            _input = torch.cat([_input, c.value], -2)
        stats = self.conv1(_input)
        stats = self.conv2(
            self.activation(
              self.norm(stats)
            )
        )
        if self.mean_only:
            m, logs = stats, torch.zeros_like(stats)
        else:
            m, logs = stats.chunk(2, -2)
            if self.scale_range is not None:
                _max, _min = self.scale_range
                logs = torch.sigmoid(logs) * (_max - _min) + _min
                logs = torch.log(logs)
        x1 = m + x1 * torch.exp(logs)
        ret = torch.cat([x0, x1], -2)
        logs = TensorMask.use_mask(logs.transpose(-1, -2), x.tensor.mask)
        logdet = x.logdet + logs
        return TensorLogdet(
            TensorMask(ret, x.tensor.mask, axis=x.tensor.axis),
            logdet
        )

    def reverse(self, x: TensorMask,
                c: Optional[TensorMask] = None) -> TensorMask:
        x0, x1 = x.value.chunk(2, -2)
        _input = x0
        if c is not None and self.condition_dim is not None:
            _input = torch.cat([x0, c.value], -2)
        stats = self.linear1(_input)
        stats = self.linear2(
            self.activation(
              self.norm(stats)
            )
        )
        if self.mean_only:
            m, logs = stats, torch.zeros_like(stats)
        else:
            m, logs = stats.chunk(2, -2)
            if self.scale_range is not None:
                _max, _min = self.scale_range
                logs = torch.sigmoid(logs) * (_max - _min) + _min
                logs = torch.log(logs)
        x1 = (x1 - m) * torch.exp(-logs)
        if self.flip:
            x0, x1 = x1, x0
        ret = torch.cat([x0, x1], -2)
        return TensorMask(ret, x.mask, axis=x.axis)


class CouplingStack(nn.Module):
    def __init__(self, dim: int, hp: Hparams,
                 condition_dim: Optional[int] = None
                 ) -> None:
        super().__init__()
        hp.check_arg_in_hparams("num_layers",
                                "layer")
        assert hp.num_layers % 2 == 0
        identifier = hp.get("identifier", "LinearCoupling")
        if identifier == "RationalQuadraticSplineCoupling":
            module = RationalQuadraticSplineCoupling
        elif identifier == "LinearCoupling":
            module = LinearCoupling
        elif identifier == "ConvCoupling":
            module = ConvCoupling
        else:
            raise ValueError(f"{hp.identifier} is not supported")
        self.condition_dim = condition_dim
        self.dim = dim
        self.layers = nn.ModuleList([
            module(dim, True, hp.layer,
                   condition_dim=condition_dim)
            for i in range(hp.num_layers)
        ])
        self.identifier = identifier

    def forward(self, x: TensorLogdet,
                c: Optional[TensorMask] = None) -> TensorLogdet:
        if self.identifier == "ConvCoupling":
            x = TensorLogdet(x.tensor.transpose(), x.logdet)
            c = c.transpose()
        for layer in self.layers:
            x = layer(x, c=c)
        if self.identifier == "ConvCoupling":
            x = TensorLogdet(x.tensor.transpose(), x.logdet)
        return x

    def reverse(self, x: TensorMask,
                c: Optional[TensorMask] = None) -> TensorMask:
        if self.identifier == "ConvCoupling":
            x = x.transpose()
            c = c.transpose()
        for layer in reversed(self.layers):
            x = layer.reverse(x, c=c)
        if self.identifier == "ConvCoupling":
            x = x.transpose()
        return x
