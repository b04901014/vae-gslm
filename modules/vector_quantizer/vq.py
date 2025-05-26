import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams.hp import Hparams
import vector_quantize_pytorch
from typing import Union, NamedTuple, Optional
from utils.tensormask import TensorMask
from utils.attr import AttrDict
from training_lib.losses import masked_loss


class VQOutput(NamedTuple):
    quantized: Union[TensorMask, torch.Tensor]
    indices: torch.Tensor
    loss: Optional[torch.Tensor]


class VectorQuantizer(nn.Module):
    def __init__(self, hp: Hparams) -> None:
        super().__init__()
        model = getattr(vector_quantize_pytorch,
                        hp.identifier, None)
        if model is None:
            raise AttributeError(f"{hp.identifier} is not recognized.")
        dictionary = hp.to_dict()
        del dictionary["identifier"]
        self.model = model(**dictionary)

    def forward(self, x: Union[TensorMask, torch.Tensor]) -> VQOutput:
        mask = None
        if isinstance(x, TensorMask):
            mask = x.mask
            x = x.value
        output = list(self.model(x))
        if mask is not None:
            output[0] = TensorMask(output[0], mask).apply_mask()
        if len(output) == 2:
            output.append(None)
        if output[1].dim == 4:
            output[1] = output[1].permute([1, 2, 3, 0]).flatten(2, 3)
        output = VQOutput(output[0], output[1], output[2])
        return output


class SimpleVectorQuantizer(nn.Module):
    def __init__(self, dim: int,
                 codebook_size: int,
                 codebook_loss_weight: float,
                 commit_loss_weight: float
                 ) -> None:
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebooks = nn.Parameter(
            torch.rand(codebook_size, dim) * 2 - 1.
        )
        self.codebook_loss_weight = codebook_loss_weight
        self.commit_loss_weight = commit_loss_weight

    def forward(self, x: TensorMask) -> AttrDict:
        mask = x.mask
        x = x.value
        b, t, c = x.size()
        x_pow = x.pow(2).sum(-1)  # b, t
        x_pow = x_pow[..., None]
        c_pow = self.codebooks.pow(2).sum(-1)  # k
        c_pow = c_pow[None, None].expand(b, t, -1)
        xc = x @ self.codebooks.T  # b, t, k
        distance = (x_pow + c_pow - 2 * xc).sqrt()
        ind = torch.argmin(distance, dim=-1)  # b, t
        codebook_quantized = self.codebooks[ind]
        quantized = (codebook_quantized - x).detach() + x
        commit_loss = (codebook_quantized.detach() - x).pow(2).mean(-1)
        commit_loss = commit_loss * self.commit_loss_weight
        commit_loss = TensorMask(commit_loss[..., None], mask)
        codebook_loss = (codebook_quantized - x.detach()).pow(2).mean(-1)
        codebook_loss = codebook_loss * self.codebook_loss_weight
        codebook_loss = TensorMask(codebook_loss[..., None], mask)
        loss = masked_loss(commit_loss,
                           codebook_loss,
                           fn=lambda x, y: x + y)
        return AttrDict(
            indicies=TensorMask(ind, mask).apply_mask(),
            output=TensorMask(quantized, mask).apply_mask(),
            loss=loss
        )

    def get_output(self, ind: torch.Tensor):
        return self.codebooks[ind]


class SimpleBestRQ(nn.Module):
    def __init__(self, dim: int,
                 codebook_size: int,
                 ) -> None:
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        codebooks = torch.randn(codebook_size, dim)
        projection = torch.empty(dim, dim)
        nn.init.xavier_normal_(projection)
        self.register_buffer('projection', projection)
        self.register_buffer('codebooks', codebooks)

    def forward(self, x: TensorMask) -> TensorMask:
        mask = x.mask
        x = x.value
        b, t, c = x.size()
        x = x @ self.projection
        x = F.normalize(x, dim=-1)
        x_pow = x.pow(2).sum(-1)  # b, t
        x_pow = x_pow[..., None]
        codes = F.normalize(self.codebooks, dim=-1)
        c_pow = codes.pow(2).sum(-1)  # k
        c_pow = c_pow[None, None].expand(b, t, -1)
        xc = x @ codes.T  # b, t, k
        distance = (x_pow + c_pow - 2 * xc).sqrt()
        ind = torch.argmin(distance, dim=-1)  # b, t
        return TensorMask(ind, mask).apply_mask()
