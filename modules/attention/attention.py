import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams.hp import Hparams
from utils.tensormask import TensorMask
from utils.helpers import make_padding_mask
import math
from typing import Optional, Mapping, Any, Tuple
from modules.position.embedding import get_positional_encoding


def reshape_head(q, k, v, num_heads):
    def _reshape(x):
        b, t, c = x.size()
        x = x.view(b, t, num_heads, c // num_heads)
        x = x.transpose(1, 2)
        return x
    return _reshape(q), _reshape(k), _reshape(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, hp: Hparams) -> None:
        super().__init__()
        hp.check_arg_in_hparams("nheads", "causal")
        self.hp = hp
        self.nheads = hp.nheads
        self.dim = dim
        assert self.dim % self.nheads == 0
        self.head_dim = self.dim // self.nheads
        self.in_proj = nn.Linear(dim, dim * 3,
                                 bias=hp.get("bias", None))
        self.out_proj = nn.Linear(self.dim, self.dim,
                                  bias=hp.get("bias", None))
        self.dropout_p = hp.get("dropout", 0.0)

    def forward(self, x: TensorMask,
                rpe_pair: Optional[Tuple[str, Any]] = None,
                rpe_bias: Optional[torch.Tensor] = None,
                return_attn: bool = False,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                return_kv: bool = False,
                ) -> Mapping[str, Any]:
        """
        Return attention weights are only for debugging
        purpose, will slow speed.
        """
        outputs = dict()
        if rpe_pair is not None:
            rpe_id, rpe = rpe_pair
        else:
            rpe_id, rpe = None, None
        q, k, v = self.in_proj(x.value).chunk(3, -1)
        if rpe_id in ["SinCos", "Rotary"]:
            q, k = rpe(q), rpe(k)
        kv_mask = x.mask
        if past_kv is not None:
            k = torch.cat([past_kv["key"], k], 1)
            v = torch.cat([past_kv["value"], v], 1)
            kv_mask = TensorMask(k).mask
        attn = make_padding_mask(kv_mask, kv_mask)
        if self.hp.causal:
            causal_mask = torch.ones_like(attn).tril(diagonal=0)
            attn = attn & causal_mask
        attn = (torch.zeros_like(attn, dtype=v.dtype).
                masked_fill_(~attn, float("-inf")))
        attn = attn[:, None].expand(-1, self.nheads, -1, -1)
        if rpe_id in ["ALiBi", "T5RPE"]:
            bias = rpe(attn)[None]
            attn = attn + bias
            outputs['rpe_bias'] = bias
        if rpe_bias is not None:
            attn = attn + rpe_bias
        attn = attn[:, :, -x.value.shape[1]:]
        _q, _k, _v = reshape_head(q, k, v, self.nheads)
        output = F.scaled_dot_product_attention(_q, _k, _v,
                                                attn_mask=attn,
                                                dropout_p=self.dropout_p)
        output = output.transpose(1, 2).reshape(q.size(0), q.size(1), self.dim)
        output = self.out_proj(output)
        outputs['output'] = TensorMask(output, x.mask).apply_mask()
        if return_kv:
            outputs["kv"] = {
                "key": k.detach(),
                "value": v.detach()
            }
        if return_attn:
            with torch.cuda.amp.autocast(dtype=torch.float32):
                _q, _k, attn = _q.float(), _k.float(), attn.float()
                attn_weights = torch.softmax((_q @ _k.transpose(-2, -1) /
                                              math.sqrt(_q.size(-1))) + attn,
                                             dim=-1)
            outputs["attn"] = attn_weights.detach()
        return outputs

    def custom_weight_init(self, init_std: float):
        std = init_std / math.sqrt(self.dim / 3)
        self.in_proj.weight.data.uniform_(-std, std)
        self.out_proj.weight.data.uniform_(-std, std)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, hp: Hparams) -> None:
        super().__init__()
        hp.check_arg_in_hparams("nheads")
        self.hp = hp
        self.nheads = hp.nheads
        self.dim = dim
        assert self.dim % self.nheads == 0
        self.head_dim = self.dim // self.nheads
        self.kv_proj = nn.Linear(dim, dim * 2,
                                 bias=hp.get("bias", None))
        self.q_proj = nn.Linear(dim, dim,
                                bias=hp.get("bias", None))
        self.out_proj = nn.Linear(self.dim, self.dim,
                                  bias=hp.get("bias", None))
        self.dropout_p = hp.get("dropout", 0.0)
        self.rpe_id, self.rpe, self.rpe_target = None, None, None
        if self.hp.has("rpe"):
            self.rpe_id = self.hp.rpe.identifier
            assert self.rpe_id in ["SinCos", "Rotary"]
            self.rpe = get_positional_encoding(self.rpe_id,
                                               self.hp.rpe,
                                               self.dim,
                                               self.nheads)
            self.rpe_target = self.hp.rpe.get("target", None)

    def forward(self, q: TensorMask,
                kv: TensorMask,
                return_attn: bool = False,
                ) -> Mapping[str, Any]:
        """
        Return attention weights are only for debugging
        purpose, will slow speed.
        """
        attn = make_padding_mask(q.mask, kv.mask)
        q_mask = q.mask
        q = self.q_proj(q.value)
        k, v = self.kv_proj(kv.value).chunk(2, -1)
        if self.rpe_id in ["SinCos", "Rotary"]:
            target = self.rpe_target
            if target == "source":
                q = self.rpe(q)
            elif target == "memory":
                k = self.rpe(k)
            else:
                q, k = self.rpe(q), self.rpe(k)
        attn = (torch.zeros_like(attn, dtype=v.dtype).
                masked_fill_(~attn, float("-inf")))
        attn = attn[:, None].expand(-1, self.nheads, -1, -1)
        _q, _k, _v = reshape_head(q, k, v, self.nheads)
        output = F.scaled_dot_product_attention(_q, _k, _v,
                                                attn_mask=attn,
                                                dropout_p=self.dropout_p)
        output = output.transpose(1, 2).reshape(q.size(0), q.size(1), self.dim)
        output = self.out_proj(output)
        output = {
            "output": TensorMask(output, q_mask).apply_mask(),
        }
        if return_attn:
            with torch.cuda.amp.autocast(dtype=torch.float32):
                _q, _k, attn = _q.float(), _k.float(), attn.float()
                attn_weights = torch.softmax((_q @ _k.transpose(-2, -1) /
                                              math.sqrt(_q.size(-1))) + attn,
                                             dim=-1)
            output["attn"] = attn_weights.detach()
        return output

    def custom_weight_init(self, init_std: float):
        std = init_std / math.sqrt(self.dim / 3)
        self.q_proj.weight.data.uniform_(-std, std)
        self.kv_proj.weight.data.uniform_(-std, std)
        self.out_proj.weight.data.uniform_(-std, std)
