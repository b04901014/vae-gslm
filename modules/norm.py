import torch.nn as nn
from hparams.hp import Hparams
import torch


def get_norm_fn(dim, hp: Hparams) -> nn.Module:
    if hp.identifier == "LayerNorm":
        return nn.LayerNorm(dim, eps=hp.eps)
    elif hp.identifier == "GroupNorm":
        return nn.GroupNorm(hp.num_groups, dim, eps=hp.eps)
    elif hp.identifier == "RMSNorm":
        return RMSNorm(dim, eps=hp.eps)
    elif hp.identifier == "InstanceNorm":
        return InstanceNorm(dim, eps=hp.eps)
    elif hp.identifier == "Identity":
        return nn.Identity()
    else:
        raise ValueError("{hp.identifier} not in the usable "
                         "normalization function lists.")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x.float()
        norm = x.pow(2).mean(-1)
        x_normed = x * torch.rsqrt(norm[..., None] + self.eps)
        return self.scale * x_normed


class InstanceNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Asserts B, C, T."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = x.float()
        var, mean = torch.var_mean(x, dim=1, keepdim=True)
        x_normed = (x - mean) * torch.rsqrt(var + self.eps)
        return self.weight[..., None] * x_normed + self.bias[..., None]
