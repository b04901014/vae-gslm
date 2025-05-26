import torch
import torch.nn as nn
import numpy as np


class SinCos(nn.Module):
    def __init__(self,
                 ndim: int,
                 maxpos: int = 10000,
                 fixed_pos: bool = False,
                 scaled: bool = False):
        super().__init__()
        p = torch.zeros((maxpos, ndim))
        pi = torch.arange(start=0, end=maxpos).float().unsqueeze(1)
        pi = pi * torch.exp(
            torch.arange(start=0, end=ndim, step=2).float() *
            -(np.log(10000.0) / ndim)
        )
        p[:, 0::2] = torch.sin(pi)
        p[:, 1::2] = torch.cos(pi)
        self.register_buffer('p', p, persistent=False)
        self.scalar = nn.Parameter(torch.FloatTensor([1.0])) if scaled else 1.0
        self.fixed_pos = fixed_pos

    def forward(self, x):
        if not self.fixed_pos:
            B, L, C = x.size()
            p = self.p[: L]
        else:
            p = self.p
        p = p.unsqueeze(0).expand(x.size(0), -1, -1)
        x = x + self.scalar * p
        return x

    def get(self, x: torch.LongTensor):
        return self.p[x]
