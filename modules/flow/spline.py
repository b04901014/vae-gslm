import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.tensormask import TensorMask
from typing import Optional
from hparams.hp import Hparams
from .utils import TensorLogdet
from modules.activations import get_activation
from modules.norm import get_norm_fn
import math


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


class RationalQuadraticSplineCoupling(nn.Module):
    def __init__(self,
                 dim: int,
                 flip: bool,
                 hp: Hparams,
                 condition_dim: Optional[int] = None):
        super().__init__()
        hp.check_arg_in_hparams("hidden_dim",
                                "activation",
                                "num_bins",
                                "tail_bound",
                                "norm")
        self.min_bin_width = hp.get('min_bin_width', 1e-3)
        self.min_bin_height = hp.get('min_bin_height', 1e-3)
        self.min_bin_derivative = hp.get('min_bin_derivative', 1e-3)
        self.condition_dim = condition_dim
        self.num_bins = hp.num_bins
        self.hidden_dim = hp.hidden_dim
        _condition_dim = 0
        if self.condition_dim is not None:
            _condition_dim = self.condition_dim
        self.linear1 = nn.Linear(dim // 2 + _condition_dim,
                                 hp.hidden_dim,
                                 bias=hp.get("bias", False))
        self.linear2 = nn.Linear(hp.hidden_dim,
                                 (self.num_bins * 3 - 1) * (dim // 2),
                                 bias=hp.get("bias", True))
        self.norm = get_norm_fn(hp.hidden_dim, hp.norm)
        self.activation = get_activation(hp.activation)
        self.flip = flip
        self.tail_bound = hp.tail_bound
        self.dim = dim

    def forward(self, x: TensorLogdet,
                c: Optional[TensorMask] = None) -> TensorLogdet:
        x0, x1 = x.tensor.value.chunk(2, -1)
        if self.flip:
            x0, x1 = x1, x0
        _input = x0
        if c is not None and self.condition_dim is not None:
            _input = torch.cat([x0, c.value], -1)
        stats = self.linear1(_input)
        stats = self.linear2(
            self.activation(
              self.norm(stats)
            )
        )
        b, t, n, c = stats.size()
        stats = stats.reshape(b, t, n,
                              self.dim // 2, self.num_bins * 3 - 1)
        w, h, d = stats.chunk(3, -1)
        x1, logdet = self.rational_quadratic_spline(x1,
                                                    w, h, d,
                                                    inverse=False)
        ret = torch.cat([x0, x1], -1)
        logdet = TensorMask.use_mask(logdet, x.tensor.mask)
        logdet = x.logdet + logdet
        return TensorLogdet(
            TensorMask(ret, x.tensor.mask, axis=x.tensor.axis),
            logdet
        )

    def reverse(self, x: TensorMask,
                c: Optional[TensorMask] = None) -> TensorMask:
        x0, x1 = x.value.chunk(2, -1)
        _input = x0
        if c is not None and self.condition_dim is not None:
            _input = torch.cat([x0, c.value], -1)
        stats = self.linear1(_input)
        stats = self.linear2(
            self.activation(
              self.norm(stats)
            )
        )
        b, t, n, c = stats.size()
        stats = stats.reshape(b, t, n,
                              self.dim // 2, self.num_bins * 3 - 1)
        w, h, d = stats.chunk(3, -1)
        x1, logdet = self.rational_quadratic_spline(x1,
                                                    w, h, d,
                                                    inverse=True)
        if self.flip:
            x0, x1 = x1, x0
        ret = torch.cat([x0, x1], -1)
        return TensorMask(ret, x.mask, axis=x.axis)

    def rational_quadratic_spline(
        self,
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=False
    ):
        left, bottom = -self.tail_bound, -self.tail_bound
        right, top = self.tail_bound, self.tail_bound
        sqrt_dim = math.sqrt(self.hidden_dim)
        unnormalized_widths = unnormalized_widths / sqrt_dim
        unnormalized_heights = unnormalized_heights / sqrt_dim
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = math.log(math.exp(1 - self.min_bin_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant
        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = self.min_bin_width + (
            (1 - self.min_bin_width * self.num_bins) * widths)
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
        cumwidths = (right - left) * cumwidths + left
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        derivatives = self.min_bin_derivative
        derivatives += F.softplus(unnormalized_derivatives)

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = self.min_bin_height + (
            (1 - self.min_bin_height * self.num_bins) * heights)
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
        cumheights = (top - bottom) * cumheights + bottom
        cumheights[..., 0] = bottom
        cumheights[..., -1] = top
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        if inverse:
            bin_idx = searchsorted(cumheights, inputs)[..., None]
        else:
            bin_idx = searchsorted(cumwidths, inputs)[..., None]
        bin_idx = torch.clamp(bin_idx, min=0, max=self.num_bins-1)

        input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

        input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
        delta = heights / widths
        input_delta = delta.gather(-1, bin_idx)[..., 0]

        input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
        input_derivatives_plus_one = (
            derivatives[..., 1:].gather(-1, bin_idx)[..., 0])

        input_heights = heights.gather(-1, bin_idx)[..., 0]

        if inverse:
            a = (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one
                - 2 * input_delta
            ) + input_heights * (input_delta - input_derivatives)
            _b = (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one
                - 2 * input_delta)
            b = input_heights * input_derivatives - _b
            c = -input_delta * (inputs - input_cumheights)

            discriminant = b.pow(2) - 4 * a * c
            root = (2 * c) / (-b - torch.sqrt(discriminant))
            outputs = root * input_bin_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one
                 - 2 * input_delta) * theta_one_minus_theta
            )
            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * root.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - root).pow(2)
            )
            logabsdet = torch.log(derivative_numerator)
            logabsdet = logabsdet - 2 * torch.log(denominator)
            logabsdet = -logabsdet
        else:
            theta = (inputs - input_cumwidths) / input_bin_widths
            theta_one_minus_theta = theta * (1 - theta)

            numerator = input_heights * (
                input_delta * theta.pow(2) +
                input_derivatives * theta_one_minus_theta
            )
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one
                 - 2 * input_delta) * theta_one_minus_theta
            )
            outputs = input_cumheights + numerator / denominator

            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * theta.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - theta).pow(2)
            )
            logabsdet = torch.log(derivative_numerator)
            logabsdet = logabsdet - 2 * torch.log(denominator)
        int_mask = (inputs >= -self.tail_bound) & (inputs <= self.tail_bound)
        outputs = torch.where(int_mask, outputs, inputs)
        logabsdet = torch.where(int_mask, logabsdet, 0.0)
        return outputs, logabsdet
