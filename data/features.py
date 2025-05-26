import torch
import torch.nn as nn
import abc
from utils.tensormask import TensorMask
from typing import Optional, Callable
from hparams.hp import Hparams
from torchaudio.transforms import MelSpectrogram


class FeatureProcessor(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class of all the FeatureProcessor objects.
    """

    @property
    @abc.abstractmethod
    def sample_rate(self) -> float:
        """The sample rate of the resulting feature."""

    @property
    @abc.abstractmethod
    def sample_ratio(self) -> float:
        """
        The ratio of the sample rate between the feature
        and the input signal.
        """

    @abc.abstractmethod
    def encode_single(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Encode a unbatched input signal into the feature.
        This is used specifically for dataloaders where
        the data is encoded per element.
        """

    @abc.abstractmethod
    def encode(self, signal: TensorMask) -> TensorMask:
        """Encode the input signal into the feature."""

    @abc.abstractmethod
    def decode(self, feature: TensorMask) -> TensorMask:
        """Decode the feature back to the signal."""


class MelSpecFeatureProcessor(FeatureProcessor):
    def __init__(self,
                 hp: Hparams,
                 vocoder_fn: Optional[Callable] = None,
                 ):
        """
        A wrapper for `torchaudio.transforms.MelSpectrogram`
        """
        super().__init__()
        self.hp = hp
        hp.check_arg_in_hparams("sample_rate",
                                "n_fft",
                                "hop_length",
                                "n_mels",
                                "power")
        self._sample_rate = hp.sample_rate
        self._hop_length = hp.hop_length
        self.vocoder_fn = vocoder_fn
        self.log_scale = hp.get("log_scale", True)
        win_length = hp.get("win_length", None)
        f_min = hp.get("f_min", 0.0)
        f_max = hp.get("f_max", None)
        self.transform = MelSpectrogram(sample_rate=hp.sample_rate,
                                        n_fft=hp.n_fft,
                                        win_length=win_length,
                                        hop_length=hp.hop_length,
                                        f_min=f_min,
                                        f_max=f_max,
                                        n_mels=hp.n_mels,
                                        power=hp.power,
                                        center=True)

    @property
    def sample_rate(self) -> float:
        return float(self._sample_rate) / float(self._hop_length)

    @property
    def sample_ratio(self) -> float:
        return 1.0 / float(self._hop_length)

    def _log_scale(self, signal: torch.Tensor,
                   gaurd: float = 1e-6) -> torch.Tensor:
        return torch.log(torch.clamp(signal, min=gaurd))

    def encode_single(self, signal: torch.Tensor) -> torch.Tensor:
        output = self.transform(signal).T
        if self.log_scale:
            output = self._log_scale(output)
        return output

    def encode(self, signal: TensorMask) -> TensorMask:
        output = self.transform(signal.value).transpose(-1, -2)
        if self.log_scale:
            output = self._log_scale(output)
        new_length = TensorMask.resize_length(signal.length, self.sample_ratio)
        return TensorMask.fromlength(output, new_length)

    def decode(self, feature: TensorMask) -> TensorMask:
        if self.vocoder is None:
            raise NotImplementedError("This method is not supported"
                                      " without a vocoder.")
        return self.vocoder_fn(feature)
