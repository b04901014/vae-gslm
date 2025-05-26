from __future__ import annotations
import torch
from typing import Optional, Union, List, Tuple
from lightning.fabric.utilities.types import _DEVICE


class TensorMask(object):
    def __init__(self,
                 x: torch.Tensor,
                 mask: Optional[torch.BoolTensor] = None,
                 axis: int = 1
                 ) -> None:
        """
        An object that binds a tensor with its masking.
        The dimension to be masked is always the second dimension.
        The mask is a BoolTensor with False as element to be masked.
        If the mask is not passed, it is assumed to be unmasked.
        Boudary Masks can be used to define multiple segments.
        """
        self.value = x
        if mask is None:
            mask = torch.ones([x.shape[0], x.shape[1]],
                              device=x.device, dtype=torch.bool)
        self.mask = mask
        self.axis = axis
        assert self.mask.dim() == 2
        assert axis in [1, 2], "Only Support B T ..., B C T"
        if axis == 1:
            assert x.shape[:2] == mask.shape
        else:
            assert [x.shape[0], x.shape[2]] == list(mask.shape)

    def __len__(self):
        return len(self.value)

    def __repr__(self):
        d = {
            'value': self.value,
            'mask': self.mask,
            'axis': self.axis
        }
        return repr(d)

    @classmethod
    def fromlength(cls,
                   x: torch.Tensor,
                   length: torch.LongTensor,
                   axis: int = 1
                   ) -> None:
        """Create Sequence object from length instead of mask. """
        mask = (torch.arange(x.shape[axis],
                             device=x.device)[None, :]
                < length[:, None])
        return cls(x, mask, axis)

    @classmethod
    def use_mask(cls,
                 x: torch.Tensor,
                 mask: torch.Tensor,
                 mask_value: float = 0) -> torch.tensor:
        return TensorMask(x, mask).apply_mask().value

    def apply_mask(self, mask_value: float = 0) -> TensorMask:
        assert self.axis == 1
        mask = self.mask[[...] + [None] * (self.value.dim() - self.mask.dim())]
        value = torch.where(mask, self.value, mask_value)
        return TensorMask(value, self.mask)

    def flatten(self) -> TensorMask:
        assert self.axis == 1
        b, t = self.value.shape[0], self.value.shape[1]
        value = self.value.reshape([b, t, -1])
        return TensorMask(value, self.mask)

    def transpose(self, a: int = -1, b: int = -2) -> TensorMask:
        value = self.value.transpose(a, b)
        return TensorMask(value, self.mask, axis=-self.axis + 3)

    def tolist(self, detach: bool = True) -> List[torch.Tensor]:
        assert self.axis == 1
        output = []
        for value, mask in zip(self.value, self.mask):
            v = value[mask]
            if detach:
                v = v.detach()
            output.append(v)
        return output

    def cuda(self) -> TensorMask:
        return TensorMask(self.value.cuda(), self.mask.cuda())

    def to(self, device: _DEVICE, non_blocking: bool = False) -> TensorMask:
        return TensorMask(
            self.value.to(device, non_blocking=non_blocking),
            self.mask.to(device, non_blocking=non_blocking)
        )

    def detach(self) -> TensorMask:
        return TensorMask(
            self.value.detach(),
            self.mask.detach()
        )

    def push(self, tm: Union[torch.Tensor, TensorMask]) -> TensorMask:
        assert self.axis == 1
        if isinstance(tm, torch.Tensor):
            tm = TensorMask(tm)
        return TensorMask(
            torch.cat([tm.value, self.value], 1),
            torch.cat([tm.mask, self.mask], 1)
        )

    def append(self, tm: Union[torch.Tensor, TensorMask]) -> TensorMask:
        assert self.axis == 1
        if isinstance(tm, torch.Tensor):
            tm = TensorMask(tm)
        return TensorMask(
            torch.cat([self.value, tm.value], 1),
            torch.cat([self.mask, tm.mask], 1)
        )

    def pop(self, n: Union[int, torch.Tensor] = 1) -> TensorMask:
        assert self.axis == 1
        new_length = self.length - n
        return TensorMask.fromlength(
            self.value[:, :-n], new_length
        )

    def pop_left(self, n: Union[int, torch.Tensor] = 1) -> TensorMask:
        new_length = self.length - n
        return TensorMask.fromlength(
            self.value[:, n:], new_length
        )

    def mean(self) -> torch.Tensor:
        assert self.axis == 1
        x = self.flatten().apply_mask()
        x = x / x.value.size(-1)
        x = x.value.sum() / x.length.sum()
        return x

    def abs(self) -> TensorMask:
        return TensorMask(torch.abs(self.value), self.mask)

    def size(self, i: Optional[int] = None):
        if i is not None:
            return self.value.size(i)
        return self.value.size()

    def cat(self,
            other: Union[torch.Tensor, TensorMask]
            ) -> TensorMask:
        mask = self.mask
        if isinstance(other, TensorMask):
            other = other.value
        return TensorMask(torch.cat([self.value, other], -self.axis + 3),
                          mask, axis=self.axis)

    @property
    def length(self) -> torch.Tensor:
        return self.mask.long().sum(-1)

    @property
    def device(self):
        return self.value.device

    @classmethod
    def resize_length(cls,
                      length: torch.LongTensor,
                      ratio: Union[float, torch.FloatTensor]
                      ) -> torch.LongTensor:
        return torch.ceil(length.float() * ratio).long()

    def __truediv__(self, other: Union[torch.Tensor, float, TensorMask]):
        if isinstance(other, TensorMask):
            other = other.value
        return TensorMask(self.value / other, self.mask, axis=self.axis)

    def __mul__(self, other: Union[torch.Tensor, float, TensorMask]):
        if isinstance(other, TensorMask):
            other = other.value
        return TensorMask(self.value * other, self.mask, axis=self.axis)

    def __add__(self, other: Union[torch.Tensor, float, TensorMask]):
        if isinstance(other, TensorMask):
            other = other.value
        return TensorMask(self.value + other, self.mask, axis=self.axis)

    def __sub__(self, other: Union[torch.Tensor, float, TensorMask]):
        if isinstance(other, TensorMask):
            other = other.value
        return TensorMask(self.value - other, self.mask, axis=self.axis)

    def batch_time_shuffle(self) -> TensorMask:
        """
        This operation returns a TensorMask that each unmasked parts are
        randomly shuffled along both batch and time dimension.
        """
        assert self.axis == 1 and len(self.size()) == 3
        b, t = self.size()[:2]
        index_map = torch.arange(b * t, device=self.device).reshape([b, t])
        index_map = index_map[self.mask]
        perm = torch.randperm(len(index_map), device=self.device)
        index_map = index_map[perm]
        value = torch.zeros_like(self.value).reshape([b*t, -1])
        masked_value = self.value[self.mask]
        value[index_map] = masked_value
        value = value.reshape([b, t, -1])
        return TensorMask(value, self.mask).apply_mask()

    def squeeze(self, dim: Optional[int] = None) -> TensorMask:
        if dim is not None:
            value = self.value.squeeze(dim)
        else:
            value = self.value.squeeze()
        return TensorMask(value, self.mask)

    def split(self, n: int) -> Tuple[TensorMask, TensorMask]:
        a = TensorMask(self.value[..., :n], self.mask)
        b = TensorMask(self.value[..., n:], self.mask)
        return a, b

    def expand(self) -> TensorMask:
        value = self.value[..., None]
        return TensorMask(value, self.mask)

    def long(self) -> TensorMask:
        return TensorMask(self.value.long(), self.mask)
