from typing import NamedTuple, Union
from utils.tensormask import TensorMask
import torch


class TensorLogdet(NamedTuple):
    tensor: Union[TensorMask, torch.Tensor]
    logdet: Union[float, torch.Tensor]
