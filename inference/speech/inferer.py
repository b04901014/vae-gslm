from hparams.hp import Hparams
import os
import torchaudio
from inference.inferer import BaseInferer
from trainers.speech.sampler import ARTRSampler
from data.dataset import MelSpecDataset
from models.vocoder.vocoder import HiFiGAN
from modules.diffusion.ddpm import GaussianDiffusion1D
from utils.tensormask import TensorMask


class SpeechInferer(BaseInferer):
    def __init__(self, hp: Hparams):
        super().__init__(hp)
        if self.hp_model.training.has("mel_rescale"):
            self.mel_rescale = self.hp_model.training.mel_rescale
        self.vocoder = HiFiGAN.from_pretrained(self.hp_model.vocoder.path,
                                               hp_rescale=self.mel_rescale)
        self.load_model(input_dim=self.vocoder.hp.n_mels)
        self.sampler = ARTRSampler(self.model)
        if hp.has('diffusion'):
            assert isinstance(self.model.decoder, GaussianDiffusion1D)
            if hp.diffusion.has('sampling_timesteps'):
                self.model.decoder.sampling_timesteps = (
                    hp.diffusion.sampling_timesteps)
            if hp.diffusion.has('ddim_sampling_eta'):
                self.model.decoder.ddim_sampling_eta = (
                    hp.diffusion.ddim_sampling_eta)

    def on_test_start(self):
        self.sampled = 0

    def on_test_end(self):
        self.sampled = 0

    def test_dataloader(self):
        dataset = MelSpecDataset(self.hp.data,
                                 self.vocoder.hp,
                                 self.mel_rescale)
        self.mel_sample_rate = dataset.melspec.sample_rate
        return self.get_dataloader(self.hp.data, dataset)

    def test_step(self, batch, batch_idx):
        prior = batch['mel'].value
        prior_length = int(self.hp.sample_prior_length *
                           self.mel_sample_rate)
        prior = prior[:, :prior_length]
        prior_decoded = self.vocoder.decode(TensorMask(prior))
        length = int(self.hp.sample_length *
                     self.mel_sample_rate *
                     self.model.sample_ratio)
        samples = self.sampler(
            length, prior,
            temperature=self.hp.temperature,
            truncated_norm=self.hp.get('truncated_norm', None),
            encoder_temperature=self.hp.get('encoder_temperature', 1.0)
        )
        sampled_audio = self.vocoder.decode(samples["output"])
        for audio, prior_d in zip(sampled_audio.tolist(),
                                  prior_decoded.tolist()):
            self.sampled += 1
            fn = os.path.join(self.hp.output_dir, f"{self.sampled}.wav")
            torchaudio.save(fn, audio.float().cpu()[None],
                            self.hp.data.sample_rate)
            fn = os.path.join(self.hp.output_dir, f"{self.sampled}_ov.wav")
            torchaudio.save(fn, prior_d.float().cpu()[None],
                            self.hp.data.sample_rate)
