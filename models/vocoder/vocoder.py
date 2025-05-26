from __future__ import annotations
from hparams.hp import Hparams
import torch.nn as nn
import abc
from utils.tensormask import TensorMask
import os
import torch
from models.vocoder.hfgan import Generator
from models.vocoder.hubert import HuBERT
from models.speech.soundstream import SoundStream
from typing import Optional


class Vocoder(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class of all the Vocoder objects.
    """

    @abc.abstractmethod
    def match_spec(self, hp: Hparams) -> bool:
        """
        Whether the specified hyperparms match the
        pretrained loaded spec.
        """

    @abc.abstractmethod
    def decode(self, signal: TensorMask) -> TensorMask:
        """Decode to speech signal."""

    @classmethod
    @abc.abstractmethod
    def from_pretrained(cls, path) -> Vocoder:
        """Initiate the instance with a pretrained model."""


class HiFiGAN(Vocoder):
    def __init__(self, hp: Hparams,
                 hp_rescale: Optional[Hparams] = None):
        super().__init__()
        self.hp = hp.feature
        self.hp_rescale = hp_rescale
        self.model = Generator(hp.model.generator)

    def match_spec(self, hp: Hparams) -> bool:
        return hp == self.hp

    def decode(self, signal: TensorMask) -> TensorMask:
        if self.hp_rescale is not None:
            signal = TensorMask(
                signal.value * self.hp_rescale.std + self.hp_rescale.mean,
                signal.mask
            ).apply_mask()
        return self.model(signal).apply_mask()

    @classmethod
    def from_pretrained(cls, path, **kwargs) -> HiFiGAN:
        hp = Hparams.from_yamlfile(
            os.path.join(path, 'hp.yaml')
        )
        hp.check_arg_in_hparams("model", "feature")
        hp.model.check_arg_in_hparams("generator")
        model = cls(hp, **kwargs)
        model.model.load_state_dict(
            torch.load(os.path.join(path, 'last-cpt.ckpt'))
        )
        model.model.remove_weight_norm()
        model.model.eval()
        return model


class SoundStreamIO(Vocoder):
    def __init__(self, hp: Hparams,
                 hp_rescale: Optional[Hparams] = None):
        super().__init__()
        self.vocoder = HiFiGAN.from_pretrained(hp.vocoder.path,
                                               hp_rescale=hp_rescale)
        self.hp = self.vocoder.hp
        self.model = SoundStream(hp.model, self.hp.n_mels)

    def match_spec(self, hp: Hparams) -> bool:
        return hp == self.hp

    def decode_mel(self, signal: TensorMask) -> TensorMask:
        codes = self.model.quantizer.model.get_output_from_indices(
            signal.value)
        codes = TensorMask(codes, signal.mask).apply_mask()
        mel = self.model.decoder(codes).apply_mask()
        return mel

    def decode(self, signal: TensorMask) -> TensorMask:
        mel = self.decode_mel(signal)
        return self.vocoder.decode(mel)

    @classmethod
    def from_pretrained(cls, path, **kwargs) -> HiFiGAN:
        hp = Hparams.from_yamlfile(
            os.path.join(path, 'hp.yaml')
        )
        hp.check_arg_in_hparams("model", "vocoder")
        model = cls(hp, **kwargs)
        model.model.load_state_dict(
            torch.load(os.path.join(path, 'last-cpt.ckpt'))
        )
        model.model.eval()
        return model

    def encode_mel(self, mel: TensorMask) -> TensorMask:
        vq = self.model.quantizer(self.model.encoder(mel))
        return TensorMask(vq.indices, vq.quantized.mask)

    @property
    def sample_ratio(self) -> float:
        return self.model.sample_ratio


class HuBERTIO(Vocoder):
    def __init__(self, hp: Hparams,
                 hp_rescale: Optional[Hparams] = None):
        super().__init__()
        self.vocoder = HiFiGAN.from_pretrained(hp.vocoder.path,
                                               hp_rescale=hp_rescale)
        self.hp = self.vocoder.hp
        self.model = HuBERT(hp.model,
                            self.hp.n_mels,
                            self.hp.sample_rate / self.hp.hop_length)
        self.hp_vq = Hparams(
            num_quantizers=1,
            codebook_size=hp.model.hubert.vocab_size,
            dim=hp.model.embedding_dim
        )

    def match_spec(self, hp: Hparams) -> bool:
        return hp == self.hp

    def decode_mel(self, signal: TensorMask) -> TensorMask:
        pass

    def decode(self, signal: TensorMask,
               spkr: Optional[TensorMask] = None,
               f0: Optional[TensorMask] = None) -> TensorMask:
        output = self.model.decode(
            self.model.encode(signal, spkr, f0)
        )
        return self.vocoder.decode(output)

    @classmethod
    def from_pretrained(cls, path, **kwargs) -> HiFiGAN:
        hp = Hparams.from_yamlfile(
            os.path.join(path, 'hp.yaml')
        )
        hp.check_arg_in_hparams("model", "vocoder")
        model = cls(hp, **kwargs)
        model.model.load_state_dict(
            torch.load(os.path.join(path, 'last-cpt.ckpt'),
                       map_location='cpu')
        )
        model.model.eval()
        return model

    def encode_mel(self, mel: TensorMask) -> TensorMask:
        '''A dummy function that does nothing. Since we hope
        that the discrete tokens are preprocessed instead of doing
        it on the fly.
        '''
        return mel

    @property
    def sample_ratio(self) -> float:
        return self.model.sample_ratio
