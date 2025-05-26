from models.speech.lvtr import LVTR
from data.symbols import Symbols
import torch.nn as nn
from utils.tensormask import TensorMask
from modules.linear.layers import (Embedding, LinearLayerStack,
                                   TimeAggregation,
                                   FiLM)
from modules.conv.layers import BottleNeckResNet
from modules.transformer.layers import TransformerLayerStack
from hparams.hp import Hparams
from typing import Optional, Mapping, List
import torch


class LVTTS(LVTR):
    def __init__(self, hp: Hparams,
                 symbols: Symbols,
                 input_dim: Optional[int] = None) -> None:
        hp.check_arg_in_hparams("text")
        hp.text.check_arg_in_hparams("embedding_dim",
                                     "encoder")
        hp.check_arg_in_hparams("eos", "spkr")
        hp.spkr.check_arg_in_hparams("embedding_dim")
        hp.transformer.layer.check_arg_in_hparams("cross_attn")
        super().__init__(hp, input_dim, hp.text.encoder.layer.dim)
        self.text_encoder = nn.Sequential(
            Embedding(symbols.num_symbols,
                      hp.text.embedding_dim,
                      padding_idx=symbols.pad_idx),
            TransformerLayerStack(hp.text.encoder,
                                  input_dim=hp.text.embedding_dim)
        )
        self.eos_head = LinearLayerStack(
            hp.eos,
            input_dim=hp.transformer.layer.dim,
            output_dim=1
        )
        self.spkr_encoder = nn.Sequential(
            BottleNeckResNet(hp.spkr,
                             input_dim=input_dim,
                             output_dim=hp.spkr.embedding_dim),
            TimeAggregation()
        )
        self.spkr_film = FiLM(hp.transformer.layer.dim, bias=False,
                              time_first=True,
                              in_dim=hp.spkr.embedding_dim)

    def forward(self,
                x: TensorMask,
                text: TensorMask,
                spkr: TensorMask,
                ) -> Mapping[str, TensorMask]:
        text, spkr = self.encode_condition(text, spkr)
        output = super().forward(x, text, spkr)
        eos = self.eos_head(output['transformer_latent'])
        output['eos'] = eos.squeeze(-1)
        output['condition'] = text
        return output

    def is_eos(self, x: TensorMask,
               threshold: float) -> torch.BoolTensor:
        eos = self.eos_head(x).squeeze(-1)
        return torch.sigmoid(eos.value) > threshold

    def step(self,
             x: torch.Tensor,
             c: Optional[TensorMask] = None,
             spkr: Optional[torch.Tensor] = None,
             past_kv: Optional[List] = None,
             temperature: float = 1.0,
             eos_threshold: float = 0.5,
             return_attn: bool = False) -> Mapping:
        outputs = super().step(x, c, spkr, past_kv, temperature, return_attn)
        outputs["eos"] = self.is_eos(outputs["transformer_latent"],
                                     eos_threshold).squeeze(1)
        return outputs

    def encode_condition(self,
                         text: TensorMask,
                         spkr: TensorMask,
                         return_attn: bool = False
                         ):
        spkr = self.spkr_encoder(spkr)
        text = self.text_encoder[0](text)
        text = self.text_encoder[1].run(text, return_attn=return_attn)
        if return_attn:
            attn = text['self_attn']
            return (text['output'], attn), spkr
        return text['output'], spkr
