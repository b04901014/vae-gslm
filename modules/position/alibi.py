import math
import torch
import torch.nn as nn


class ALiBi(nn.Module):
    def __init__(self, nheads: int, maxpos: int = 10000) -> None:
        super().__init__()
        context_position = torch.arange(maxpos)[:, None]
        memory_position = torch.arange(maxpos)[None, :]
        relative_position = memory_position - context_position
        relative_position = (torch.abs(relative_position).unsqueeze(0).
                             expand(nheads, -1, -1))
        slopes = torch.Tensor(self.get_slopes(nheads)) * -1
        alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
        alibi = alibi.view(nheads, maxpos, maxpos)
        self.register_buffer('alibi', alibi, persistent=False)

    def get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return (get_slopes_power_of_2(closest_power_of_2) +
                    (self.get_slopes(2*closest_power_of_2)[0::2]
                     [:n-closest_power_of_2]))

    def forward(self, x):
        return self.alibi[:, :x.size(2), :x.size(3)]
