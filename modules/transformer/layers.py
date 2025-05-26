import torch.nn as nn
import torch
from modules.attention.attention import SelfAttention, CrossAttention
from hparams.hp import Hparams
from modules.activations import get_activation
from modules.norm import get_norm_fn
from utils.tensormask import TensorMask
from typing import Optional, Mapping, List, Any, Tuple
from modules.position.embedding import get_positional_encoding
import math


class TransformerLayer(nn.Module):
    def __init__(self, hp: Hparams) -> None:
        super().__init__()
        hp.check_arg_in_hparams("ffd_size",
                                "norm",
                                "activation",
                                "dim",
                                "self_attn")
        dropout = hp.get("dropout", 0.0)
        self.hp = hp
        self.preln = hp.get('preln', True)
        self.self_attn = SelfAttention(hp.dim, hp.self_attn)
        self.cross_attn = None
        if hp.has("cross_attn"):
            self.cross_attn = CrossAttention(hp.dim, hp.cross_attn)
            self.norm2 = get_norm_fn(hp.dim, hp.norm)
            self.dropout2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hp.dim, hp.ffd_size,
                                 bias=hp.get('bias', True))
        self.linear2 = nn.Linear(hp.ffd_size, hp.dim,
                                 bias=hp.get('bias', True))

        self.norm1 = get_norm_fn(hp.dim, hp.norm)
        self.norm3 = get_norm_fn(hp.dim, hp.norm)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = get_activation(hp.activation)

    def forward(self,
                tgt: TensorMask,
                memory: Optional[TensorMask] = None,
                rpe_pair: Optional[Tuple[str, Any]] = None,
                rpe_bias: Optional[torch.Tensor] = None,
                past_kv: Optional[Mapping] = None,
                return_attn: bool = False,
                return_kv: bool = False
                ) -> Mapping:
        output = dict()
        mask = tgt.mask
        if self.preln:
            n_tgt = TensorMask(self.norm1(tgt.value),
                               mask).apply_mask()
        else:
            n_tgt = tgt
        sa = self.self_attn(n_tgt, past_kv=past_kv,
                            rpe_pair=rpe_pair,
                            rpe_bias=rpe_bias,
                            return_attn=return_attn,
                            return_kv=return_kv)
        if 'rpe_bias' in sa:
            output['rpe_bias'] = sa['rpe_bias']
        tgt = tgt.value + self.dropout1(sa['output'].value)
        if not self.preln:
            tgt = self.norm1(tgt)
        if self.cross_attn is not None:
            if self.preln:
                n_tgt = self.norm2(tgt)
            else:
                n_tgt = tgt
            n_tgt = TensorMask(n_tgt, mask).apply_mask()
            ca = self.cross_attn(n_tgt, memory,
                                 return_attn=return_attn)
            tgt = tgt + self.dropout2(ca['output'].value)
            if not self.preln:
                tgt = self.norm2(tgt)
        if self.preln:
            n_tgt = self.norm3(tgt)
        else:
            n_tgt = tgt
        tgt2 = self.linear2(self.activation(self.linear1(n_tgt)))
        tgt = tgt + self.dropout3(tgt2)
        if not self.preln:
            tgt = self.norm3(tgt)
        output["output"] = TensorMask(tgt, mask).apply_mask()
        if return_attn:
            output["self_attn"] = sa["attn"]
            if self.cross_attn is not None:
                output["cross_attn"] = ca["attn"]
        if return_kv:
            output["kv"] = sa["kv"]
        return output


class TransformerLayerStack(nn.Module):
    def __init__(self, hp: Hparams,
                 input_dim: Optional[int] = None,
                 output_dim: Optional[int] = None,
                 memory_dim: Optional[int] = None) -> None:
        super().__init__()
        hp.check_arg_in_hparams("num_layers",
                                "layer")
        self.layers = nn.ModuleList([
            TransformerLayer(hp.layer) for _ in range(hp.num_layers)
        ])
        self.out, self.linear, self.memory_linear = None, None, None
        if input_dim is not None:
            self.linear = nn.Linear(input_dim, hp.layer.dim,
                                    bias=hp.get("bias", True))
        self.is_cross_attn = False
        if hp.layer.has("cross_attn"):
            self.is_cross_attn = True
        if hp.layer.has("cross_attn") and memory_dim is not None:
            self.memory_linear = nn.Linear(memory_dim, hp.layer.dim,
                                           bias=hp.get("bias", True))
        if output_dim is not None:
            self.out = nn.Linear(hp.layer.dim, output_dim,
                                 bias=hp.get("bias", True))
        self.final_norm, self.first_norm = None, None
        if hp.get("final_ln", True):
            self.final_norm = get_norm_fn(hp.layer.dim, hp.layer.norm)
        if hp.get("first_ln", False):
            self.first_norm = get_norm_fn(hp.layer.dim, hp.layer.norm)
        self.hp = hp
        self.rpe, self.rpe_id = None, None
        if self.hp.get("rpe", False):
            self.rpe_id = self.hp.rpe.identifier
            self.rpe = get_positional_encoding(self.rpe_id,
                                               self.hp.rpe,
                                               self.hp.layer.dim,
                                               self.hp.layer.self_attn.nheads)

    def run(self,
            tgt: TensorMask,
            memory: Optional[TensorMask] = None,
            past_kv: Optional[List] = None,
            return_attn: bool = False,
            return_kv: bool = False
            ) -> Mapping[str, Any]:
        outputs = {'output': []}
        if return_attn:
            outputs['self_attn'] = []
            if self.is_cross_attn:
                outputs['cross_attn'] = []
        if return_kv:
            outputs['kv'] = []
        if past_kv is None:
            past_kv = [None for _ in range(len(self.layers))]
        output = tgt
        if self.linear is not None:
            output = TensorMask(self.linear(output.value),
                                output.mask)
            output = output.apply_mask()
        if self.first_norm is not None:
            output = TensorMask(self.first_norm(output.value),
                                output.mask).apply_mask()
        if self.memory_linear is not None and memory is not None:
            memory = TensorMask(
                self.memory_linear(memory.value),
                memory.mask)
            memory = memory.apply_mask()
        rpe_pair = (self.rpe_id, self.rpe)
        rpe_bias = None
        output_layers = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory,
                         rpe_pair=rpe_pair,
                         rpe_bias=rpe_bias,
                         past_kv=past_kv[i],
                         return_attn=return_attn,
                         return_kv=return_kv)
            if "rpe_bias" in output:
                rpe_pair = None
                rpe_bias = output["rpe_bias"]
            if return_attn:
                outputs['self_attn'].append(output['self_attn'].
                                            detach())
                if self.is_cross_attn:
                    outputs['cross_attn'].append(output['cross_attn'].
                                                 detach())
            if return_kv:
                outputs["kv"].append(output["kv"])
            output = output['output']
            output_layers.append(output)
        if self.final_norm is not None:
            output = TensorMask(self.final_norm(output.value),
                                output.mask)
            output_layers.append(output)
        if self.out is not None:
            output = TensorMask(self.out(output.value), output.mask)
            output = output.apply_mask()
        outputs['output'] = output
        outputs['layers'] = output_layers
        return outputs

    def forward(self, tgt: TensorMask,
                memory: Optional[TensorMask] = None) -> TensorMask:
        return self.run(tgt, memory=memory)['output']

    def custom_weight_init(self, init_std: float):
        std = init_std / math.sqrt(self.hp.layer.dim / 3)
        if self.rpe_id == "T5RPE":
            self.rpe.relative_attention_bias.weight.data.uniform_(-std, std)
