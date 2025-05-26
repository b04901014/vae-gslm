from hparams.hp import Hparams
import os
import torch
import torchaudio
from inference.inferer import BaseInferer
from trainers.speech.sampler import ARTRSampler
from data.dataset import DiscreteTokenDataset
from models.vocoder.vocoder import HuBERTIO
from utils.tensormask import TensorMask


class SpeechInferer(BaseInferer):
    def __init__(self, hp: Hparams):
        super().__init__(hp)
        if self.hp_model.training.has("mel_rescale"):
            self.mel_rescale = self.hp_model.training.mel_rescale
        self.hp_model.hubert.check_arg_in_hparams("path")
        soundstream = HuBERTIO.from_pretrained(
            self.hp_model.hubert.path,
            hp_rescale=self.mel_rescale)
        self.deduplicate = soundstream.model.deduplicate
        self.load_model(hp_vq=soundstream.hp_vq,
                        input_dim=soundstream.hp.n_mels)
        self.model.set_soundstream(soundstream)
        self.sampler = ARTRSampler(self.model)

    def on_test_start(self):
        self.sampled = 0

    def on_test_end(self):
        self.sampled = 0

    def test_dataloader(self):
        dataset = DiscreteTokenDataset(
            self.hp.data,
            self.model.soundstream.hp,
            self.model.soundstream.model.hp.hubert,
            self.mel_rescale
        )
        self.token_sample_rate = dataset.token_sample_rate
        return self.get_dataloader(self.hp.data, dataset)

    def test_step(self, batch, batch_idx):
        if self.deduplicate:
            prior = batch['dedup_tokens']
            min_length = torch.min(prior.length)
            prior = prior.value[:, :min_length]
            prior_length = self.hp.sample_prior_tokens
            length = self.hp.sample_tokens
        else:
            prior = batch['tokens'].value
            prior_length = int(self.hp.sample_prior_length *
                               self.token_sample_rate)
            length = int(self.hp.sample_length *
                         self.token_sample_rate)
        prior = prior[:, :prior_length]
        if self.model.soundstream.model.hp.has("spkr"):
            prior_decoded = self.model.soundstream.decode(
                TensorMask(prior), spkr=batch['cropped_mel'])
            samples = self.sampler(
                length, prior,
                temperature=self.hp.temperature,
                spkr=batch['cropped_mel']
            )
        else:
            prior_decoded = self.model.soundstream.decode(TensorMask(prior))
            samples = self.sampler(
                length, prior,
                temperature=self.hp.temperature
            )
        sampled_audio = samples["output"]
        for audio, prior_d in zip(sampled_audio.tolist(),
                                  prior_decoded.tolist()):
            self.sampled += 1
            fn = os.path.join(self.hp.output_dir, f"{self.sampled}.wav")
            torchaudio.save(fn, audio.float().cpu()[None],
                            self.hp.data.sample_rate)
            fn = os.path.join(self.hp.output_dir, f"{self.sampled}_ov.wav")
            torchaudio.save(fn, prior_d.float().cpu()[None],
                            self.hp.data.sample_rate)
