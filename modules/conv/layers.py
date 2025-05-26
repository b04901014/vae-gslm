import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams.hp import Hparams
from utils.helpers import get_padding
from modules.norm import get_norm_fn
from modules.activations import get_activation
from modules.linear.layers import FiLM
from utils.tensormask import TensorMask
from typing import Optional


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        self.two_side_padding = None
        if 'padding' in kwargs:
            padding = kwargs['padding']
            if isinstance(padding, tuple):
                assert len(padding) == 2
                kwargs['padding'] = 0
                self.two_side_padding = padding
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.two_side_padding is not None:
            padding_mode = self.padding_mode
            if padding_mode == 'zeros':
                padding_mode = 'constant'
            a, b = self.two_side_padding
            input = F.pad(input, [a, b, 0, 0, 0, 0])
        return super().forward(input)


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        self.two_side_padding = None
        assert 'output_padding' not in kwargs, "Not supported."
        if 'padding' in kwargs:
            padding = kwargs['padding']
            if isinstance(padding, tuple):
                assert len(padding) == 2
                kwargs['padding'] = 0
                self.two_side_padding = padding
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.conv_transpose1d(
            input, self.weight, self.bias, self.stride, self.padding,
            0, self.groups, self.dilation)
        if self.two_side_padding is not None:
            a, b = self.two_side_padding
            output = output[..., a:]
            if b > 0:
                output = output[..., : -b]
        return output


class LayerScale(nn.Module):
    def __init__(self, dim: float, eps: float, axis: int):
        super().__init__()
        dims = [1, 1, 1]
        dims[axis] = dim
        self.gamma = nn.Parameter(eps * torch.ones(dims),
                                  requires_grad=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.gamma * input


class ResidualBlock(nn.Module):
    def __init__(self, hp: Hparams):
        super().__init__()
        hp.check_arg_in_hparams("in_channels",
                                "hidden_channels",
                                "kernel_size",
                                "norm",
                                "activation")
        aux_in_channels = hp.get("aux_in_channels", 0)
        assert hp.norm.identifier != "LayerNorm", "BCT format not supported"
        causal_padding = hp.get("causal_padding", False)
        future_padding = hp.get("future_padding", False)
        padding = get_padding(hp.kernel_size, causal=causal_padding,
                              future=future_padding)
        self.norm = get_norm_fn(hp.in_channels, hp.norm)
        self.act = get_activation(hp.activation)
        self.conv1 = Conv1d(hp.in_channels, hp.in_channels,
                            kernel_size=hp.kernel_size,
                            padding=padding,
                            groups=hp.in_channels)
        self.conv2 = nn.Conv1d(hp.in_channels + aux_in_channels,
                               hp.hidden_channels,
                               kernel_size=1,
                               padding=0)
        self.conv3 = nn.Conv1d(hp.hidden_channels, hp.in_channels,
                               kernel_size=1,
                               padding=0)

        self.dropout = nn.Dropout(hp.get("dropout", 0.0))
        use_shortcut = hp.get("shortcut", False)
        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv1d(hp.in_channels, hp.in_channels,
                          kernel_size=1,
                          padding=0),
                get_activation(hp.activation)
            )
        else:
            self.shortcut = nn.Identity()
        if hp.has("layer_scale"):
            hp.layer_scale.check_arg_in_hparams("eps")
            self.layer_scale = LayerScale(hp.in_channels,
                                          hp.layer_scale.eps,
                                          axis=1)
        else:
            self.layer_scale = nn.Identity()

    def forward(self, x: TensorMask) -> TensorMask:
        """x: BCT TensorMask."""
        h = self.act(
            self.conv2(
                self.norm(
                    self.conv1(
                        x.value
                    )
                )
            )
        )
        h = self.dropout(
            self.conv3(
                h
            )
        )
        h = self.layer_scale(h)
        shortcut = self.shortcut(x.value)
        return TensorMask(h + shortcut, x.mask, axis=2)


class Upsample(nn.Module):
    def __init__(self, n_channels: int,
                 kernel_size: int,
                 stride: int,
                 norm_hp: Hparams,
                 causal_padding: bool = False,
                 future_padding: bool = False,
                 out_channels: int = None):
        super().__init__()
        self.stride = stride
        if out_channels is None:
            out_channels = n_channels
        padding = get_padding(kernel_size, stride=stride,
                              causal=causal_padding,
                              future=future_padding)
        self.norm = get_norm_fn(n_channels, norm_hp)
        self.conv = ConvTranspose1d(n_channels, out_channels,
                                    kernel_size,
                                    stride,
                                    padding=padding)

    def forward(self, x: TensorMask) -> TensorMask:
        """x: BCT TensorMask."""
        length = TensorMask.resize_length(x.length,
                                          float(self.stride))
        return TensorMask.fromlength(self.conv(self.norm(x.value)), length,
                                     axis=2)


class Downsample(nn.Module):
    def __init__(self, n_channels: int,
                 kernel_size: int,
                 stride: int,
                 norm_hp: Hparams,
                 causal_padding: bool = False,
                 future_padding: bool = False,
                 out_channels: int = None):
        super().__init__()
        self.stride = stride
        if out_channels is None:
            out_channels = n_channels
        padding = get_padding(kernel_size, stride=stride,
                              causal=causal_padding,
                              future=future_padding)
        self.norm = get_norm_fn(n_channels, norm_hp)
        self.conv = Conv1d(n_channels, out_channels,
                           kernel_size,
                           stride,
                           padding=padding)

    def forward(self, x: TensorMask) -> TensorMask:
        """x: BCT TensorMask."""
        length = TensorMask.resize_length(x.length,
                                          1.0 / float(self.stride))
        return TensorMask.fromlength(self.conv(self.norm(x.value)), length,
                                     axis=2)


class ConditionalResidualBlock(ResidualBlock):
    def __init__(self, hp: Hparams):
        self.condition_type = hp.get("condition_type", "film")
        if self.condition_type == "film":
            super().__init__(hp)
            self.film = FiLM(hp.in_channels, time_first=False,
                             in_dim=hp.get("in_dim", None))
        else:
            hp.aux_in_channels = hp.get("in_dim", hp.in_channels)
            super().__init__(hp)

    def forward(self, x: TensorMask, c: TensorMask) -> TensorMask:
        """x: BCT TensorMask."""
        if self.condition_type == "film":
            h = self.film(
                    self.norm(
                        self.conv1(
                            x.value
                        )
                    ), c
                )
        else:
            h = self.norm(self.conv1(x.value))
            h = torch.cat([h, c.value], 1)
        h = self.act(self.conv2(h))
        h = self.dropout(
            self.conv3(
                h
            )
        )
        h = self.layer_scale(h)
        shortcut = self.shortcut(x.value)
        return TensorMask(h + shortcut, x.mask, axis=2)


class TemporalResidualBlock(ResidualBlock):
    def __init__(self, hp: Hparams):
        super().__init__(hp)
        hp.check_arg_in_hparams("time_dim")
        self.time_emb = nn.Linear(hp.time_dim, hp.in_channels)

    def forward(self, x: TensorMask, c: torch.Tensor) -> TensorMask:
        """x: BCT TensorMask.  c: BC Tensor"""
        t = self.time_emb(self.act(c))[..., None]
        h = self.act(
            self.conv2(
                self.norm(
                    self.conv1(
                        x.value
                    ) + t
                )
            )
        )
        h = self.dropout(
            self.conv3(
                h
            )
        )
        h = self.layer_scale(h)
        shortcut = self.shortcut(x.value)
        return TensorMask(h + shortcut, x.mask, axis=2)


class TCResidualBlock(ResidualBlock):
    def __init__(self, hp: Hparams):
        self.condition_type = hp.get("condition_type", "film")
        if self.condition_type == "film":
            super().__init__(hp)
            self.film = FiLM(hp.in_channels, time_first=False,
                             in_dim=hp.get("in_dim", None))
        else:
            hp.aux_in_channels = hp.get("in_dim", hp.in_channels)
            super().__init__(hp)
        hp.check_arg_in_hparams("time_dim")
        self.time_emb = nn.Linear(hp.time_dim, hp.in_channels)

    def forward(self, x: TensorMask,
                c: TensorMask, t: torch.Tensor) -> TensorMask:
        """x: BCT TensorMask."""
        t = self.time_emb(self.act(t))[..., None]
        if self.condition_type == "film":
            h = self.film(
                    self.norm(
                        self.conv1(
                            x.value
                        ) + t
                    ), c
                )
        else:
            h = self.norm(self.conv1(x.value) + t)
            h = torch.cat([h, c.value], 1)
        h = self.act(self.conv2(h))
        h = self.dropout(
            self.conv3(
                h
            )
        )
        h = self.layer_scale(h)
        shortcut = self.shortcut(x.value)
        return TensorMask(h + shortcut, x.mask, axis=2)


class ResNet(nn.Module):
    def __init__(self, hp: Hparams,
                 input_dim: Optional[int] = None,
                 output_dim: Optional[int] = None,
                 conditional: bool = False) -> None:
        super().__init__()
        self.hp = hp
        causal_padding = hp.layer.get("causal_padding", False)
        hp.check_arg_in_hparams("num_layers",
                                "layer")
        resample_rates = hp.get("resample_rates",
                                [1 for _ in range(hp.num_layers)])
        resample_ksize = hp.get("resample_ksize",
                                [3 for _ in range(hp.num_layers)])
        assert len(resample_rates) == hp.num_layers
        self.layers = nn.ModuleList([
            ConditionalResidualBlock(hp.layer)
            if conditional else ResidualBlock(hp.layer)
            for _ in range(hp.num_layers)
        ])
        samples = []
        for rk_size, rate in zip(resample_ksize, resample_rates):
            assert isinstance(rate, int) and rate != 0
            if rate in [1, -1]:
                samples.append(nn.Identity())
            elif rate > 1:
                samples.append(Upsample(hp.layer.in_channels,
                                        rk_size,
                                        rate,
                                        hp.layer.norm,
                                        causal_padding=causal_padding))
            else:
                samples.append(Downsample(hp.layer.in_channels,
                                          rk_size,
                                          -rate,
                                          hp.layer.norm,
                                          causal_padding=causal_padding))
        self.samples = nn.ModuleList(samples)
        self.linear = None
        if input_dim is not None:
            self.linear = nn.Linear(input_dim, hp.layer.in_channels)
        self.out_linear, self.final_norm, self.first_norm = None, None, None
        if output_dim is not None:
            self.out_linear = nn.Linear(hp.layer.in_channels, output_dim)
        if hp.get('final_norm', False):
            self.final_norm = get_norm_fn(hp.layer.in_channels, hp.layer.norm)
        if hp.get('first_norm', False):
            self.first_norm = get_norm_fn(hp.layer.in_channels, hp.layer.norm)
        self.conditional = conditional

    def forward(self, x: TensorMask,
                c: Optional[TensorMask] = None) -> TensorMask:
        """x: BTC TensorMask."""
        if self.linear is not None:
            x = TensorMask(self.linear(x.value), x.mask).apply_mask()
        x = x.transpose()
        if self.first_norm is not None:
            x = TensorMask(self.first_norm(x.value),
                           x.mask, axis=2)
        if c is not None:
            c = c.transpose()
        for sample, layer in zip(self.samples, self.layers):
            if self.conditional:
                x = sample(layer(x, c))
            else:
                x = sample(layer(x))
        if self.final_norm is not None:
            x = TensorMask(self.final_norm(x.value),
                           x.mask, axis=2)
        x = x.transpose()
        if self.out_linear is not None:
            x = TensorMask(self.out_linear(x.value),
                           x.mask).apply_mask()
        return x.apply_mask()

    @property
    def sample_ratio(self) -> float:
        ret = 1.0
        resample_rates = self.hp.get("resample_rates",
                                     [1 for _ in range(self.hp.num_layers)])
        for rate in resample_rates:
            if rate > 0:
                ret *= rate
            else:
                ret /= -rate
        return ret


class BottleNeckResNet(nn.Module):
    def __init__(self, hp: Hparams,
                 input_dim: Optional[int] = None,
                 output_dim: Optional[int] = None) -> None:
        super().__init__()
        self.hp = hp
        hp.check_arg_in_hparams("num_layers",
                                "layer",
                                "init_channel",
                                "out_channels",
                                "hidden_channels",
                                "resample_rates",
                                "resample_ksize")
        upward_layer_boundary = 100000000000  # Random Large Number
        if hp.has("upward_layer"):
            upward_layer_boundary = hp.upward_layer.boundary
            assert upward_layer_boundary < hp.num_layers
        resample_rates = hp.resample_rates
        resample_ksize = hp.resample_ksize
        out_channels = hp.out_channels
        init_channel = hp.init_channel
        hidden_channels = hp.hidden_channels
        in_channels = ([init_channel] + out_channels)[:-1]
        if hp.has("conditional"):
            hp.check_arg_in_hparams("condition_dim")
            hp.layer.in_dim = hp.condition_dim
            if hp.has("upward_layer"):
                hp.upward_layer.in_dim = hp.condition_dim
        conditional = hp.get("conditional",
                             [False for _ in range(hp.num_layers)])
        self.time_dim = hp.get("time_dim", None)
        assert len(resample_rates) == hp.num_layers
        assert len(resample_ksize) == hp.num_layers
        assert len(out_channels) == hp.num_layers
        assert len(hidden_channels) == hp.num_layers
        layers = []
        samples = []
        skip_conv = []
        self.skip_connection = hp.get("skip_connection",
                                      [None] * hp.num_layers)
        self.skip_concat = hp.get('connection_type', None) == "concat"
        for i in range(hp.num_layers):
            if i < upward_layer_boundary:
                c_layer = hp.layer
            else:
                c_layer = hp.upward_layer
            causal_padding = c_layer.get("causal_padding", False)
            future_padding = c_layer.get("future_padding", False)
            c_layer.in_channels = in_channels[i]
            c_layer.hidden_channels = hidden_channels[i]
            c_layer.aux_in_channels = 0
            if self.skip_connection[i] is not None and self.skip_concat:
                skip_conv.append(nn.Conv1d(in_channels[i] * 2,
                                           in_channels[i],
                                           1, 1, 0))
            else:
                skip_conv.append(nn.Identity())
            if conditional[i] and (self.time_dim is not None):
                c_layer.time_dim = self.time_dim
                layers.append(TCResidualBlock(c_layer))
            elif conditional[i]:
                layers.append(ConditionalResidualBlock(c_layer))
            elif self.time_dim is not None:
                c_layer.time_dim = self.time_dim
                layers.append(TemporalResidualBlock(c_layer))
            else:
                layers.append(ResidualBlock(c_layer))
            rk_size, rate = resample_ksize[i], resample_rates[i]
            assert isinstance(rate, int) and rate != 0
            if rate in [1, -1]:
                assert in_channels[i] == out_channels[i]
                samples.append(nn.Identity())
            elif rate > 1:
                samples.append(Upsample(in_channels[i],
                                        rk_size,
                                        rate,
                                        c_layer.norm,
                                        causal_padding=causal_padding,
                                        future_padding=future_padding,
                                        out_channels=out_channels[i]))
            else:
                samples.append(Downsample(in_channels[i],
                                          rk_size,
                                          -rate,
                                          c_layer.norm,
                                          causal_padding=causal_padding,
                                          future_padding=future_padding,
                                          out_channels=out_channels[i]))
        self.layers = nn.ModuleList(layers)
        self.samples = nn.ModuleList(samples)
        self.skip_conv = nn.ModuleList(skip_conv)
        self.linear = None
        if input_dim is not None:
            self.linear = nn.Linear(input_dim, init_channel)
        self.out_linear, self.final_norm, self.first_norm = None, None, None
        if output_dim is not None:
            self.out_linear = nn.Linear(out_channels[-1], output_dim)
        if hp.get('final_norm', False):
            self.final_norm = get_norm_fn(out_channels[-1], hp.layer.norm)
        if hp.get('first_norm', False):
            self.first_norm = get_norm_fn(hp.layer.in_channels, hp.layer.norm)
        self.conditional = conditional
        assert len(self.skip_connection) == hp.num_layers

    def forward(self, x: TensorMask,
                c: Optional[TensorMask] = None,
                t: Optional[torch.Tensor] = None) -> TensorMask:
        """x: BTC TensorMask."""
        if self.linear is not None:
            x = TensorMask(self.linear(x.value), x.mask).apply_mask()
        x = x.transpose()
        if self.first_norm is not None:
            x = TensorMask(self.first_norm(x.value),
                           x.mask, axis=2)
        if c is not None:
            c = c.transpose()
        records = [x]
        for sample, layer, conditional, skip, skp in zip(self.samples,
                                                         self.layers,
                                                         self.conditional,
                                                         self.skip_connection,
                                                         self.skip_conv):
            if conditional and (self.time_dim is not None):
                x = sample(layer(x, c, t))
            elif conditional:
                x = sample(layer(x, c))
            elif self.time_dim is not None:
                x = sample(layer(x, t))
            else:
                x = sample(layer(x))
            if skip is not None:
                if not self.skip_concat:
                    x = x + records[skip]
                else:
                    x = x.cat(records[skip])
                    x = TensorMask(skp(x.value), x.mask, axis=2)
            records.append(x)
        if self.final_norm is not None:
            x = TensorMask(self.final_norm(x.value),
                           x.mask, axis=2)
        x = x.transpose()
        if self.out_linear is not None:
            x = TensorMask(self.out_linear(x.value),
                           x.mask).apply_mask()
        return x.apply_mask()

    @property
    def sample_ratio(self) -> float:
        ret = 1.0
        for rate in self.hp.resample_rates:
            if rate > 0:
                ret *= rate
            else:
                ret /= -rate
        return ret


class ConvNormAct(nn.Module):
    def __init__(self, hp: Hparams):
        super().__init__()
        hp.check_arg_in_hparams("in_channels",
                                "out_channels",
                                "kernel_size",
                                "stride",
                                "norm",
                                "activation")
        assert hp.norm.identifier != "LayerNorm", "BCT format not supported"
        causal_padding = hp.get("causal_padding", False)
        future_padding = hp.get("future_padding", False)
        padding = get_padding(hp.kernel_size, causal=causal_padding,
                              future=future_padding)
        self.norm = get_norm_fn(hp.out_channels, hp.norm)
        self.act = get_activation(hp.activation)
        if hp.stride < 0 or hp.stride == 1:
            if hp.stride < 0:
                stride = -hp.stride
            else:
                stride = hp.stride
            self.conv = Conv1d(hp.in_channels, hp.out_channels,
                               kernel_size=hp.kernel_size,
                               stride=stride,
                               padding=padding)
            self.stride = 1 / float(stride)
        else:
            stride = hp.stride
            self.conv = ConvTranspose1d(hp.in_channels, hp.out_channels,
                                        kernel_size=hp.kernel_size,
                                        stride=stride,
                                        padding=padding)
            self.stride = float(stride)
        self.dropout = nn.Dropout(hp.get("dropout", 0.0))

    def forward(self, x: TensorMask) -> TensorMask:
        """x: BCT TensorMask."""
        h = self.act(
            self.norm(
                self.conv(
                    x.value
                )
            )
        )
        h = self.dropout(h)
        if self.stride != 1:
            length = TensorMask.resize_length(x.length,
                                              1.0 / float(self.stride))
            return TensorMask.fromlength(h, length, axis=2)
        return TensorMask(h, x.mask, axis=2)


class CNNStack(nn.Module):
    def __init__(self, hp: Hparams,
                 input_dim: Optional[int] = None,
                 output_dim: Optional[int] = None) -> None:
        super().__init__()
        self.hp = hp
        hp.check_arg_in_hparams("num_layers",
                                "layer",
                                "init_channel",
                                "out_channels",
                                "resample_rates",
                                "resample_ksize")
        resample_rates = hp.resample_rates
        resample_ksize = hp.resample_ksize
        out_channels = hp.out_channels
        init_channel = hp.init_channel
        in_channels = ([init_channel] + out_channels)[:-1]
        assert len(resample_rates) == hp.num_layers
        assert len(resample_ksize) == hp.num_layers
        assert len(out_channels) == hp.num_layers
        layers = []
        for i in range(hp.num_layers):
            c_layer = hp.layer
            c_layer.in_channels = in_channels[i]
            c_layer.out_channels = out_channels[i]
            c_layer.kernel_size = resample_ksize[i]
            c_layer.stride = resample_rates[i]
            layers.append(ConvNormAct(c_layer))
        self.layers = nn.ModuleList(layers)
        self.linear = None
        if input_dim is not None:
            self.linear = nn.Linear(input_dim, init_channel)
        self.out_linear, self.final_norm, self.first_norm = None, None, None
        if output_dim is not None:
            self.out_linear = nn.Linear(out_channels[-1], output_dim)

    def forward(self, x: TensorMask) -> TensorMask:
        """x: BTC TensorMask."""
        if self.linear is not None:
            x = TensorMask(self.linear(x.value), x.mask).apply_mask()
        x = x.transpose()
        for layer in self.layers:
            x = layer(x)
        x = x.transpose()
        if self.out_linear is not None:
            x = TensorMask(self.out_linear(x.value),
                           x.mask).apply_mask()
        return x.apply_mask()

    @property
    def sample_ratio(self) -> float:
        ret = 1.0
        for rate in self.hp.resample_rates:
            if rate > 0:
                ret *= rate
            else:
                ret /= -rate
        return ret
