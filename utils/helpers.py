import torch
import torch.nn.functional as F
from collections.abc import Iterable
from typing import Mapping, Any, Optional, Union
from .tensormask import TensorMask
from lightning_utilities.core.apply_func import apply_to_collection
from lightning.fabric.utilities.apply_func import _BLOCKING_DEVICE_TYPES
from lightning.fabric.utilities.apply_func import _TransferableDataType
from lightning.fabric.utilities.types import _DEVICE
from pathlib import Path
import re
import torchaudio


def move_data_to_device(batch: Any, device: _DEVICE) -> Any:
    if isinstance(device, str):
        device = torch.device(device)

    def batch_to(data: Any) -> Any:
        non_blocking = False
        if (isinstance(data, (torch.Tensor, TensorMask)) and
                isinstance(device, torch.device) and
                device.type not in _BLOCKING_DEVICE_TYPES):
            non_blocking = True
        data_output = data.to(device, non_blocking=non_blocking)
        if data_output is not None:
            return data_output
        return data

    return apply_to_collection(batch,
                               dtype=_TransferableDataType,
                               function=batch_to)


def random_crop_1d(signal: torch.Tensor,
                   sample_rate: float,
                   min_crop_length_sec: float,
                   return_start_end: bool = False):
    """The signal is cropped on the first dimension. """
    min_crop_length = int(min_crop_length_sec * sample_rate)
    if min_crop_length >= len(signal):
        if return_start_end:
            return signal, 0, len(signal)
        return signal
    start_point = torch.randint(low=0,
                                high=len(signal)-min_crop_length+1,
                                size=())
    signal = signal[start_point: start_point+min_crop_length]
    if return_start_end:
        return signal, start_point, start_point+min_crop_length
    return signal


def pad_1d(signal: torch.Tensor,
           sample_rate: float,
           length_sec: float,
           padding_mode: str = 'constant') -> torch.Tensor:
    """The signal is padded to the specified length. """
    length = int(length_sec * sample_rate)
    if len(signal) >= length:
        return signal
    padding_idx = torch.zeros((signal.dim() * 2,), dtype=torch.long)
    padding_idx[-1] = length - len(signal)
    padding_idx = padding_idx.tolist()
    return torch.nn.functional.pad(signal,
                                   padding_idx,
                                   mode=padding_mode)


def truncate_1d(signal: torch.Tensor,
                sample_rate: float,
                length_sec: float) -> torch.Tensor:
    """The signal is truncated to the specified length. """
    length = int(length_sec * sample_rate)
    if len(signal) < length:
        return signal
    return signal[:length]


def pad_to_max_length(batch: Iterable[Mapping],
                      max_lengths: Optional[Mapping[str, int]] = None,
                      ) -> Mapping[str, Any]:
    '''
    Pad every `tensor` inside the batch to the max length
    of the element of the batch. This can be used to create
    batches.

    This function only parses tensors that have dimensions
    bigger than or equal to 1 (sequence)
    '''
    if max_lengths is None:
        max_lengths = dict()
    mlb = dict()
    for element in batch:
        for k, v in element.items():
            if isinstance(v, torch.Tensor):
                if v.dim() >= 1:
                    if k in max_lengths:
                        mlb[k] = max_lengths[k]
                    else:
                        if k not in mlb:
                            mlb[k] = 0
                        mlb[k] = max(len(v), mlb[k])
    ret = dict()
    ret_scalar = dict()
    ret_not_tensor = dict()
    for element in batch:
        for k, v in element.items():
            if isinstance(v, torch.Tensor):
                if v.dim() < 1:
                    if k not in ret_scalar:
                        ret_scalar[k] = []
                    ret_scalar.append(v)
                else:
                    if k not in ret:
                        ret[k] = []
                    if len(v) > mlb[k]:
                        v = v[:mlb[k]]
                    mask = torch.BoolTensor(
                        [True] * len(v) + [False] * (mlb[k] - len(v))
                    )
                    value = pad_1d(v, 1.0, float(mlb[k]))
                    ret[k].append((value, mask))
            else:
                if k not in ret_not_tensor:
                    ret_not_tensor[k] = []
                ret_not_tensor[k].append(v)
    for k in ret.keys():
        value, mask = zip(*ret[k])
        ret[k] = TensorMask(torch.stack(value), torch.stack(mask))
    for k in ret_scalar.keys():
        ret_scalar[k] = torch.stack(ret_scalar[k])
    ret.update(ret_not_tensor)
    ret.update(ret_scalar)
    return ret


def get_padding(kernel_size, dilation=1, stride=1,
                causal=False, future=False):
    padding = int(((kernel_size - 1) * dilation + 1 - stride) / 2)
    if causal:
        padding = (padding * 2, 0)
    elif future:
        padding = (0, padding * 2)
    return padding


def make_padding_mask(a: torch.BoolTensor,
                      b: torch.BoolTensor) -> torch.BoolTensor:
    return b[..., None, :].expand(-1, a.size(1), -1)


def get_last_ckpt(directory: str) -> str:
    ckpts = Path(directory).glob("*-cpt.ckpt")

    def get_numbers(s):
        x = re.findall(r'step=(\d+)', s.stem)
        if not x:
            raise ValueError(f"Checkpoint {s} is does not contain steps...")
        return int(x[0])
    return list(sorted(ckpts, key=get_numbers))[-1]


def interpolate(x: TensorMask,
                ratio: Union[torch.Tensor, float]) -> TensorMask:
    x_v, length = x.value, x.length
    s = int(x_v.size(1) * ratio)
    out = F.interpolate(x_v.transpose(1, 2),
                        size=s,
                        mode='linear')
    new_length = TensorMask.resize_length(length,
                                          ratio)
    out = TensorMask.fromlength(out.transpose(1, 2), new_length)
    return out


def repeat_batch(x: TensorMask, n: int) -> TensorMask:
    value, mask = x.value, x.mask
    b, t, c = value.size()
    value = value[None].expand(n, -1, -1, -1)
    mask = mask[None].expand(n, -1, -1, -1)
    mask = mask.reshape([n * b, t])
    value = value.reshape([n * b, t, c])
    return TensorMask(value, mask)


def compute_mfcc(mel: TensorMask,
                 dct: torch.Tensor,
                 delta: bool = False,
                 cmvn: bool = False) -> TensorMask:
    mfcc = torch.matmul(mel.value, dct)
    if cmvn:
        mean = torch.sum(mfcc, 1) / mel.length[..., None]
        mean = mean[:, None]
        var = torch.sum((mfcc - mean) ** 2, 1) / mel.length[..., None]
        var = var[:, None]
        mfcc = (mfcc - mean) / (var + 1e-6) ** 0.5
    if delta:
        delta_1 = torchaudio.functional.compute_deltas(
            mfcc.transpose(-1, -2)
        )
        delta_2 = torchaudio.functional.compute_deltas(
            delta_1
        )
        delta_1 = delta_1.transpose(-1, -2)
        delta_2 = delta_2.transpose(-1, -2)
        mfcc = torch.cat([mfcc, delta_1, delta_2], -1)
    return TensorMask(mfcc, mel.mask).apply_mask()


def specaug(x: TensorMask,
            feat_drop_rate: float,
            time_drop_rate: float):
    mask = torch.ones_like(x.value).bool()
    if feat_drop_rate > 0.0:
        feat_drop = torch.rand(x.value.size(0), x.value.size(1))
        feat_drop = feat_drop < feat_drop_rate
        mask = mask.transpose(1, 2)
        mask[feat_drop] = False
        mask = mask.transpose(1, 2)
    if feat_drop_rate > 0.0:
        time_drop = torch.rand(x.value.size(0), x.value.size(1))
        time_drop = time_drop < time_drop_rate
        mask[time_drop] = False
    return TensorMask(
        torch.where(mask, x.value, 0.0),
        x.mask
    )
