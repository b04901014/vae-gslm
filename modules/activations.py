import torch.nn as nn
from hparams.hp import Hparams


def get_activation(hp: Hparams) -> nn.Module:
    if hp.identifier == "ReLU":
        return nn.ReLU()
    elif hp.identifier == "SELU":
        return nn.SELU()
    elif hp.identifier == "GELU":
        return nn.GELU()
    elif hp.identifier == "LeakyRELU":
        return nn.LeakyReLU(negative_slope=hp.slope)
    elif hp.identifier == "SiLU":
        return nn.SiLU()
    else:
        raise ValueError(f"{hp.identifier} not in the usable "
                         "activation function lists.")
