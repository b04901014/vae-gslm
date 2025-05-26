import torch
from torch import nn
from modules.position.absolute import SinCos
from modules.activations import get_activation
from hparams.hp import Hparams
from modules.conv.layers import ResNet, BottleNeckResNet
from utils.tensormask import TensorMask


class TimeEmbedding(nn.Module):
    def __init__(self, hp: Hparams):
        super().__init__()
        hp.check_arg_in_hparams("activation", "maxpos", "dim")
        self.n_channels = hp.dim
        self.lin1 = nn.Linear(self.n_channels, self.n_channels,
                              bias=hp.get("bias", True))
        self.act = get_activation(hp.activation)
        self.lin2 = nn.Linear(self.n_channels, self.n_channels,
                              bias=hp.get("bias", True))
        self.embedding = SinCos(self.n_channels, maxpos=hp.maxpos)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.embedding.get(t)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


class ConditionalUNet(nn.Module):
    def __init__(self, cond_dim: int, noise_dim: int, hp: Hparams):
        super().__init__()
        hp.check_arg_in_hparams("cond_net", "unet",
                                "time_embedding")
        assert not hp.unet.has("resample_rates")
        self.cond_net = ResNet(hp.cond_net,
                               input_dim=cond_dim+hp.time_embedding.dim,
                               output_dim=hp.unet.layer.hidden_channels)
        self.time_embedding = TimeEmbedding(hp.time_embedding)
        self.noise_linear = nn.Linear(noise_dim, hp.unet.layer.in_channels)
        self.unet = ResNet(hp.unet,
                           output_dim=noise_dim,
                           conditional=True)

    def forward(self,
                noise: TensorMask,
                t: torch.Tensor,
                cond: TensorMask) -> TensorMask:
        """
        noise, cond: B, T, C
        t: B,
        """
        t = self.time_embedding(t)[:, None].expand(-1, cond.value.size(1), -1)
        cond = TensorMask(torch.cat([cond.value, t], -1),
                          cond.mask).apply_mask()
        cond = self.cond_net(cond)
        noise = TensorMask(
            self.noise_linear(noise.value),
            noise.mask
        ).apply_mask()
        return self.unet(noise, cond)

    @property
    def sample_ratio(self) -> float:
        return self.cond_net.sample_ratio


class ConditionalBottleNeckUNet(nn.Module):
    def __init__(self, cond_dim: int, noise_dim: int, hp: Hparams):
        super().__init__()
        hp.check_arg_in_hparams("unet",
                                "time_embedding")
        hp.unet.check_arg_in_hparams("conditional")
        hp.unet.time_dim = hp.time_embedding.dim
        self.cond_net = nn.Linear(cond_dim,
                                  hp.unet.condition_dim)
        self.time_embedding = TimeEmbedding(hp.time_embedding)
        self.unet = BottleNeckResNet(hp.unet,
                                     input_dim=noise_dim,
                                     output_dim=noise_dim)

    def forward(self,
                noise: TensorMask,
                t: torch.Tensor,
                cond: TensorMask) -> TensorMask:
        """
        noise, cond: B, T, C
        t: B,
        """
        t = self.time_embedding(t)
        cond = TensorMask(self.cond_net(
            cond.value
        ), cond.mask).apply_mask()
        return self.unet(noise, cond, t)
