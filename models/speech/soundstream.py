import torch.nn as nn
from hparams.hp import Hparams
from modules.conv.layers import BottleNeckResNet
from modules.vector_quantizer.vq import VectorQuantizer
from typing import Mapping, Optional
from utils.tensormask import TensorMask


class SoundStream(nn.Module):
    def __init__(self, hp: Hparams,
                 input_dim: Optional[int] = None) -> None:
        super().__init__()
        hp.check_arg_in_hparams("encoder",
                                "decoder",
                                "quantizer")
        self.hp = hp
        self.encoder = BottleNeckResNet(hp.encoder,
                                        input_dim=input_dim,
                                        output_dim=hp.quantizer.dim)
        self.quantizer = VectorQuantizer(hp.quantizer)
        self.decoder = BottleNeckResNet(hp.decoder,
                                        input_dim=hp.quantizer.dim,
                                        output_dim=input_dim)

    @property
    def sample_ratio(self) -> float:
        return self.encoder.sample_ratio

    def forward(self,
                x: TensorMask,
                ) -> Mapping[str, TensorMask]:
        z = self.encoder(x)
        vq_z = self.quantizer(z)
        rec = self.decoder(vq_z.quantized)
        return {
            'reconstruction': rec,
            'aux_loss': vq_z.loss
        }
