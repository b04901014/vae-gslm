from hparams.hp import Hparams
import os
import torchaudio
from inference.inferer import BaseInferer
from trainers.tts.sampler import ARTRTTSSampler
from data.dataset import MelSpecDataset
from models.vocoder.vocoder import HiFiGAN
from modules.diffusion.ddpm import GaussianDiffusion1D
from data.symbols import Symbols


class TTSInferer(BaseInferer):
    def __init__(self, hp: Hparams):
        super().__init__(hp)
        hp.check_arg_in_hparams("max_sample_length",
                                "min_sample_length",
                                "temperature",
                                "eos_threshold")
        if self.hp_model.training.has("mel_rescale"):
            self.mel_rescale = self.hp_model.training.mel_rescale
        self.vocoder = HiFiGAN.from_pretrained(self.hp_model.vocoder.path,
                                               hp_rescale=self.mel_rescale)
        self.symbols = Symbols.load(
            os.path.join(self.hp.ckpt_path, 'symbols.json')
        )
        self.load_model(symbols=self.symbols,
                        input_dim=self.vocoder.hp.n_mels)
        self.sampler = ARTRTTSSampler(self.model)
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
        dataset.symbols = self.symbols
        return self.get_dataloader(self.hp.data, dataset)

    def test_step(self, batch, batch_idx):
        samples = self.sampler(
            batch['text'],
            batch['cropped_mel'],
            int(self.hp.max_sample_length *
                self.mel_sample_rate *
                self.model.sample_ratio),
            int(self.hp.min_sample_length *
                self.mel_sample_rate *
                self.model.sample_ratio),
            temperature=self.hp.temperature,
            eos_threshold=self.hp.eos_threshold,
            return_attn=False
        )
        sampled_audio = self.vocoder.decode(samples["output"])
        condition_rec = self.vocoder.decode(batch['mel'])
        for audio, condition, text in zip(
            sampled_audio.tolist(),
            condition_rec.tolist(),
            batch['text_written_form']
        ):
            self.sampled += 1
            fn = os.path.join(self.hp.output_dir, f"{self.sampled}.wav")
            torchaudio.save(fn, audio.float().cpu()[None],
                            self.hp.data.sample_rate)
            fn = os.path.join(self.hp.output_dir, f"{self.sampled}_c.wav")
            torchaudio.save(fn, condition.float().cpu()[None],
                            self.hp.data.sample_rate)
            fn = os.path.join(self.hp.output_dir, f"{self.sampled}.txt")
            with open(fn, 'w') as f:
                f.write(text)
