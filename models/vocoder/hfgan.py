import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
try:
    from torch.nn.utils.parametrizations import weight_norm
    from torch.nn.utils.parametrize import (remove_parametrizations
                                            as remove_weight_norm)
except ImportError:
    from torch.nn.utils import weight_norm, remove_weight_norm
from collections.abc import Iterable
import numpy as np
from utils.tensormask import TensorMask
from utils.helpers import get_padding
from hparams.hp import Hparams
from typing import Tuple

LRELU_SLOPE = 0.1


def WNConv1d(in_channel, out_channel, kernel_size, stride,
             dilation, padding):
    return weight_norm(Conv1d(in_channel, out_channel,
                              kernel_size, stride,
                              dilation=dilation,
                              padding=padding))


def WNConv1dT(in_channel, out_channel, kernel_size, stride,
              padding, output_padding):
    return weight_norm(ConvTranspose1d(in_channel, out_channel,
                                       kernel_size, stride,
                                       padding=padding,
                                       output_padding=output_padding))


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class ResBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 dilation: Iterable[int] = (1, 3, 5),
                 weight_norm: bool = True,
                 ) -> None:
        super().__init__()
        self.weight_norm = weight_norm
        conv = WNConv1d if weight_norm else Conv1d
        self.convs1 = nn.ModuleList([
            conv(channels, channels, kernel_size, 1, dilation=dilation[0],
                 padding=get_padding(kernel_size, dilation[0])),
            conv(channels, channels, kernel_size, 1, dilation=dilation[1],
                 padding=get_padding(kernel_size, dilation[1])),
            conv(channels, channels, kernel_size, 1, dilation=dilation[2],
                 padding=get_padding(kernel_size, dilation[2])),
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            conv(channels, channels, kernel_size, 1, dilation=1,
                 padding=get_padding(kernel_size, 1)),
            conv(channels, channels, kernel_size, 1, dilation=1,
                 padding=get_padding(kernel_size, 1)),
            conv(channels, channels, kernel_size, 1, dilation=1,
                 padding=get_padding(kernel_size, 1)),
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        if not self.weight_norm:
            return
        for layer in self.convs1:
            remove_weight_norm(layer, 'weight')
        for layer in self.convs2:
            remove_weight_norm(layer, 'weight')


class Generator(nn.Module):
    def __init__(self, hp: Hparams) -> None:
        super().__init__()
        hp.check_arg_in_hparams("weight_norm",
                                "resblock_kernel_sizes",
                                "upsample_rates",
                                "in_channels",
                                "upsample_initial_channel",
                                "kernel_size",
                                "upsample_kernel_sizes",
                                "upsample_initial_channel",
                                "resblock_dilation_sizes")
        conv = WNConv1d if hp.weight_norm else Conv1d
        convT = WNConv1dT if hp.weight_norm else ConvTranspose1d
        self.hp = hp
        self.num_kernels = len(hp.resblock_kernel_sizes)
        self.num_upsamples = len(hp.upsample_rates)
        self.conv_pre = conv(hp.in_channels,
                             hp.upsample_initial_channel,
                             hp.kernel_size, 1, 1,
                             padding=get_padding(hp.kernel_size, 1))
        resblock = ResBlock

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(hp.upsample_rates,
                                       hp.upsample_kernel_sizes)):
            self.ups.append(convT(hp.upsample_initial_channel // (2 ** i),
                                  (hp.upsample_initial_channel //
                                   (2 ** (i + 1))),
                                  k, u, padding=(u // 2 + u % 2),
                                  output_padding=u % 2))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = hp.upsample_initial_channel // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(hp.resblock_kernel_sizes,
                                           hp.resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = conv(ch, 1, hp.kernel_size, 1, 1,
                              padding=get_padding(hp.kernel_size, 1))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x: TensorMask) -> TensorMask:
        total_upsample = np.prod(self.hp.upsample_rates)
        new_length = TensorMask.resize_length(x.length, total_upsample)
        x = self.conv_pre(x.value.transpose(-1, -2))
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x).squeeze(1)
        return TensorMask.fromlength(x, new_length)

    def remove_weight_norm(self):
        if not self.hp.weight_norm:
            return
        print('Removing weight norm...')
        for layer in self.ups:
            remove_weight_norm(layer, 'weight')
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre, 'weight')
        remove_weight_norm(self.conv_post, 'weight')


class DiscriminatorP(nn.Module):
    def __init__(self, period: int,
                 kernel_size: int = 5,
                 stride: int = 3,
                 use_weight_norm: bool = True) -> None:
        super().__init__()
        self.period = period
        norm_f = weight_norm if use_weight_norm else lambda x: x
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 64, (kernel_size, 1), (stride, 1),
                          padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(64, 128, (kernel_size, 1), (stride, 1),
                          padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 256, (kernel_size, 1), (stride, 1),
                          padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(256, 512, (kernel_size, 1), (stride, 1),
                          padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), 1,
                          padding=(get_padding(kernel_size), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, hp: Hparams) -> None:
        super().__init__()
        hp.check_arg_in_hparams("periods",
                                "weight_norm")
        self.hp = hp
        self.discriminators = nn.ModuleList([
            DiscriminatorP(period,
                           use_weight_norm=hp.weight_norm)
            for period in hp.periods
        ])

    def forward(self, y):
        y_d_rs = []
        fmap_rs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
        return y_d_rs, fmap_rs


class DiscriminatorS(nn.Module):
    def __init__(self, use_weight_norm: bool = True) -> None:
        super().__init__()
        norm_f = weight_norm if use_weight_norm else lambda x: x
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, hp: Hparams) -> None:
        super().__init__()
        hp.check_arg_in_hparams("num_scales",
                                "weight_norm")
        self.hp = hp
        self.discriminators = nn.ModuleList([
            DiscriminatorS(hp.weight_norm)
            for _ in range(hp.num_scales)
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2)
            for _ in range(hp.num_scales - 1)
        ])

    def forward(self, y):
        y_d_rs = []
        fmap_rs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
            y_d_r, fmap_r = d(y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
        return y_d_rs, fmap_rs


class DiscriminatorR(nn.Module):
    def __init__(self, resolution: Tuple[int],
                 use_weight_norm: bool = True) -> None:
        super().__init__()
        norm_f = weight_norm if use_weight_norm else lambda x: x
        self.resolution = resolution

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []
        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x,
                  (int((n_fft - hop_length) / 2),
                   int((n_fft - hop_length) / 2)),
                  mode='reflect')
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft,
                       hop_length=hop_length,
                       win_length=win_length,
                       center=False,
                       return_complex=True)
        mag = torch.abs(x)
        return mag


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, hp: Hparams) -> None:
        super().__init__()
        hp.check_arg_in_hparams("resolutions",
                                "weight_norm")
        self.resolutions = hp.resolutions
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(tuple(resolution), hp.weight_norm)
             for resolution in self.resolutions]
        )

    def forward(self, y):
        y_d_rs = []
        fmap_rs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
        return y_d_rs, fmap_rs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
    return loss


def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        _loss = torch.mean((1 - dg) ** 2)
        loss += _loss
    return loss
