from hparams.hp import Hparams
import torch
import numpy as np
from inference.inferer import BaseInferer
from data.dataset import MelSpecDataset, DiscreteTokenDataset
from models.vocoder.vocoder import HiFiGAN, HuBERTIO


class LikelihoodEstimator(BaseInferer):
    def __init__(self, hp: Hparams) -> None:
        super().__init__(hp)
        self.beta = hp.get("beta", None)
        if self.hp_model.training.has("mel_rescale"):
            self.mel_rescale = self.hp_model.training.mel_rescale
        if self.hp.model.identifier == "models.speech.discrete.DiscreteAR":
            self.type = 'hubert'
            self.hp_model.hubert.check_arg_in_hparams("path")
            soundstream = HuBERTIO.from_pretrained(
                self.hp_model.hubert.path,
                hp_rescale=self.mel_rescale)
            self.deduplicate = soundstream.model.deduplicate
            self.load_model(hp_vq=soundstream.hp_vq,
                            input_dim=soundstream.hp.n_mels)
            self.model.set_soundstream(soundstream)
            if not self.deduplicate:
                self.input_key = 'tokens'
            else:
                self.input_key = 'dedup_tokens'
        else:
            self.type = 'lvtr'
            _hp_beta = self.hp_model.training.get("amortized_beta", None)
            self.vocoder = HiFiGAN.from_pretrained(self.hp_model.vocoder.path,
                                                   hp_rescale=self.mel_rescale)
            self.load_model(input_dim=self.vocoder.hp.n_mels,
                            hp_beta=_hp_beta)
            self.input_key = 'mel'
        self.use_tokens = (
            hasattr(self.model, 'use_tokens') and self.model.use_tokens)
        self.gamma = 1.0
        if self.use_tokens:
            self.hp_hubert = Hparams(
                deduplicate=False,
                sample_rate=self.hp_model.hubert.sample_rate)
            self.gamma = hp.get('gamma',
                                self.hp_model.training.get(
                                    "token_kld_weight", 1.0))

    def on_test_start(self):
        self.scores = []

    def on_test_end(self):
        self.scores = np.concatenate(self.scores)

    def test_dataloader(self):
        if self.type == 'hubert':
            dataset = DiscreteTokenDataset(
                self.hp.data,
                self.model.soundstream.hp,
                self.model.soundstream.model.hp.hubert,
                self.mel_rescale
            )
            self.token_sample_rate = dataset.token_sample_rate
        else:
            if self.use_tokens:
                dataset = DiscreteTokenDataset(self.hp.data,
                                               self.vocoder.hp,
                                               self.hp_hubert,
                                               self.mel_rescale)
            else:
                dataset = MelSpecDataset(self.hp.data,
                                         self.vocoder.hp,
                                         self.mel_rescale)
            self.mel_sample_rate = dataset.melspec.sample_rate
        self.hp.data.sampler.drop_last = False
        self.hp.data.drop_last = False
        return self.get_dataloader(self.hp.data, dataset)

    def test_step(self, batch, batch_idx):
        beta = None
        f0 = None
        if self.hp.model.identifier == "models.speech.discrete.DiscreteAR":
            f0 = batch.get('f0', None)
        if self.beta is not None:
            beta = torch.full((batch[self.input_key].size(0),), self.beta,
                              dtype=batch[self.input_key].value.dtype,
                              device=batch[self.input_key].value.device)
        model_input = batch[self.input_key]
        if self.use_tokens:
            model_input = batch['tokens'].expand().cat(batch['mel'])
        score = self.model.likelihood(model_input,
                                      f0=f0,
                                      beta=beta,
                                      gamma=self.gamma)
        self.scores.append(score.cpu().numpy())
