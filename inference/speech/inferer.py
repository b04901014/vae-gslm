from hparams.hp import Hparams
import os
import torch
import torchaudio
from inference.inferer import BaseInferer
from trainers.speech.sampler import ARTRSampler
from modules.diffusion.ddpm import GaussianDiffusion1D
from utils.tensormask import TensorMask
from data.dataset import MelSpecDataset, DiscreteTokenDataset
from models.vocoder.vocoder import HiFiGAN, HuBERTIO, MixedIO


class SpeechInferer(BaseInferer):
    def __init__(self, hp: Hparams):
        super().__init__(hp)
        if self.hp_model.training.has("mel_rescale"):
            self.mel_rescale = self.hp_model.training.mel_rescale
        if self.hp.model.identifier == "models.speech.discrete.DiscreteAR":
            self.type = 'hubert'
            self.hp_model.hubert.check_arg_in_hparams("path")
            self.mixed_tokens = hp.hubert.get("mixed_tokens", False)
            if self.mixed_tokens:
                soundstream = MixedIO.from_pretrained(
                    hp.hubert.path,
                    hp_rescale=self.mel_rescale)
                self.deduplicate = False
            else:
                soundstream = HuBERTIO.from_pretrained(
                    self.hp_model.hubert.path,
                    hp_rescale=self.mel_rescale)
                self.deduplicate = soundstream.model.deduplicate
            self.load_model(hp_vq=soundstream.hp_vq,
                            input_dim=soundstream.hp.n_mels,
                            mixed_tokens=self.mixed_tokens)
            self.model.set_soundstream(soundstream)
            if not self.deduplicate:
                self.input_key = 'tokens'
            else:
                self.input_key = 'dedup_tokens'
        else:
            self.type = 'lvtr'
            self.vocoder = HiFiGAN.from_pretrained(self.hp_model.vocoder.path,
                                                   hp_rescale=self.mel_rescale)
            self.load_model(input_dim=self.vocoder.hp.n_mels)
            self.input_key = 'mel'
        self.sampler = ARTRSampler(self.model)
        self.use_tokens = (
            hasattr(self.model, 'use_tokens') and self.model.use_tokens)
        self.gamma = 1.0
        if self.use_tokens:
            self.hp_hubert = Hparams(
                deduplicate=False,
                sample_rate=self.hp_model.hubert.sample_rate)
        if hp.has('diffusion'):
            if self.hp.model.identifier == "models.speech.discrete.DiscreteAR":
                assert isinstance(self.model.soundstream.model.decoder,
                                  GaussianDiffusion1D)
                m = self.model.soundstream.model.decoder
            else:
                assert isinstance(self.model.decoder, GaussianDiffusion1D)
                m = self.model.decoder
            if hp.diffusion.has('sampling_timesteps'):
                m.sampling_timesteps = (
                    hp.diffusion.sampling_timesteps)
            if hp.diffusion.has('ddim_sampling_eta'):
                m.ddim_sampling_eta = (
                    hp.diffusion.ddim_sampling_eta)
        if self.hp.has("vad") and self.hp.vad.get("auth_token", None) is not None:
            from pyannote.audio import Model
            from pyannote.audio.pipelines import VoiceActivityDetection
            model = Model.from_pretrained(
                "pyannote/segmentation-3.0",
                use_auth_token=self.hp.vad.auth_token)
            self.pipeline = VoiceActivityDetection(segmentation=model)
            HYPER_PARAMETERS = {
                "min_duration_on": 0.0,
                "min_duration_off": 0.0
            }
            self.pipeline.instantiate(HYPER_PARAMETERS)
        self.nsteps = 0

    def on_test_start(self):
        self.sampled = 0

    def on_test_end(self):
        self.sampled = 0

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
                self.token_sample_rate = dataset.token_sample_rate
            else:
                dataset = MelSpecDataset(self.hp.data,
                                         self.vocoder.hp,
                                         self.mel_rescale)
        self.mel_sample_rate = dataset.melspec.sample_rate
        self.hp.data.sampler.drop_last = False
        self.hp.data.drop_last = False
        return self.get_dataloader(self.hp.data, dataset)

    def test_step(self, batch, batch_idx):
        if self.type == 'hubert':
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
            mel_prior = batch['mel'].value
            mel_prior_length = int(self.hp.sample_prior_length *
                                   self.mel_sample_rate)
            mel_prior = TensorMask(mel_prior[:, :mel_prior_length])
            if self.model.f0 is not None:
                prior_f0 = batch['f0'].value[:, :prior_length]
                prior = torch.cat([prior[..., None], prior_f0[..., None]], -1)
            if self.model.soundstream.model.hp.has("spkr"):
                prior_decoded = self.model.decode(
                    TensorMask(prior), spkr=mel_prior)
            else:
                prior_decoded = self.model.decode(TensorMask(prior))
            args = [length, prior]
            kwargs = {
                'temperature': self.hp.temperature,
                'return_attn': self.hp.get('plot_attn', False)
            }
            if self.model.soundstream.model.hp.has("spkr"):
                kwargs['spkr'] = mel_prior
            samples = self.sampler(*args, **kwargs)
            sampled_audio = samples["output"]
        else:
            prior = batch['mel'].value
            prior_length = int(self.hp.sample_prior_length *
                               self.mel_sample_rate)
            prior = prior[:, :prior_length]
            prior_decoded = self.vocoder.decode(TensorMask(prior))
            length = int(self.hp.sample_length *
                         self.mel_sample_rate *
                         self.model.sample_ratio)
            if self.use_tokens:
                prior = torch.cat([(batch['tokens'].value
                                   [:, :prior_length, None]),
                                   prior],
                                  -1)
            samples = self.sampler(
                length, prior,
                temperature=self.hp.temperature,
                token_temperature=self.hp.get("token_temperature", 1.0),
                truncated_norm=self.hp.get('truncated_norm', None),
                encoder_temperature=self.hp.get('encoder_temperature', 1.0)
            )
            self.nsteps += 1
            sampled_audio = self.vocoder.decode(samples["output"])
        for audio, prior_d in zip(sampled_audio.tolist(),
                                  prior_decoded.tolist()):
            self.sampled += 1
            fn = os.path.join(self.hp.output_dir, f"{self.sampled}.wav")
            uncliped_audio = audio.float().cpu()[None]
            torchaudio.save(fn, uncliped_audio,
                            self.hp.data.sample_rate)
            if self.hp.has("vad") and self.hp.vad.get("auth_token", None) is not None:
                # VAD post-process...?
                vad = self.pipeline(fn)
                start = vad._tracks.keys()[-1].start
                end = vad._tracks.keys()[-1].end
                if (end - start) < 1.5:  # trim the last segment
                    end = vad._tracks.keys()[-2].end
                end = int(end * self.hp.data.sample_rate)
                end = min(end + 4000, len(uncliped_audio[0]))
                cliped_audio = uncliped_audio[:, : end]
                torchaudio.save(fn, cliped_audio,
                                self.hp.data.sample_rate)
