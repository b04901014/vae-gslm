from models.speech.discrete import DiscreteAR
from training_lib.trainer import BaseTrainer
from training_lib.optimizer import create_optimizer
from data.dataset import MelSpecDataset, DiscreteTokenDataset
from training_lib.losses import masked_ce_loss, masked_l1_loss
from hparams.hp import Hparams
from models.vocoder.vocoder import SoundStreamIO, HuBERTIO
from .sampler import ARTRSampler
import torch
import os


class DiscreteARTrainer(BaseTrainer):
    def __init__(self, hp: Hparams) -> None:
        super().__init__(hp)
        self.hp = hp
        self.mel_rescale = None
        if hp.training.has("mel_rescale"):
            hp.training.mel_rescale.check_arg_in_hparams("mean", "std")
            self.mel_rescale = hp.training.mel_rescale
        if hp.has("soundstream"):
            self.token_type = 'soundstream'
            hp.soundstream.check_arg_in_hparams("path")
            soundstream = SoundStreamIO.from_pretrained(
                hp.soundstream.path,
                hp_rescale=self.mel_rescale)
            self.model = DiscreteAR(hp.model, soundstream.model.hp.quantizer,
                                    input_dim=soundstream.hp.n_mels)
        elif hp.has("hubert"):
            self.token_type = 'hubert'
            hp.hubert.check_arg_in_hparams("path")
            soundstream = HuBERTIO.from_pretrained(
                hp.hubert.path,
                hp_rescale=self.mel_rescale)
            self.model = DiscreteAR(hp.model, soundstream.hp_vq,
                                    input_dim=soundstream.hp.n_mels)
            self.deduplicate = soundstream.model.deduplicate

        else:
            raise ValueError("One of soundstream or hubert"
                             "should be in hparams!")
        hp.check_arg_in_hparams("logging")
        hp.logging.check_arg_in_hparams("num_samples",
                                        "temperature",
                                        "sample_length",
                                        "sample_prior_length",
                                        "plot_attn")
        self.apply(self.init_weights)
        self.model.set_soundstream(soundstream)
        self.sampler = ARTRSampler(self.model)
        self.automatic_optimization = False

    def train_dataloader(self):
        if self.token_type == 'soundstream':
            train_dataset = MelSpecDataset(self.hp.data.train,
                                           self.model.soundstream.hp,
                                           self.mel_rescale,
                                           name="train dataset")
        else:
            train_dataset = DiscreteTokenDataset(
                self.hp.data.train,
                self.model.soundstream.hp,
                self.model.soundstream.model.hp.hubert,
                self.mel_rescale,
                name="train dataset"
            )
        return self.get_dataloader(self.hp.data.train, train_dataset)

    def val_dataloader(self):
        if self.token_type == 'soundstream':
            val_dataset = MelSpecDataset(self.hp.data.val,
                                         self.model.soundstream.hp,
                                         self.mel_rescale,
                                         name="validation dataset")
        else:
            val_dataset = DiscreteTokenDataset(
                self.hp.data.val,
                self.model.soundstream.hp,
                self.model.soundstream.model.hp.hubert,
                self.mel_rescale,
                name="validation dataset"
            )
            self.val_token_sample_rate = val_dataset.token_sample_rate
        self.val_mel_sample_rate = val_dataset.melspec.sample_rate
        return self.get_dataloader(self.hp.data.val, val_dataset)

    def configure_optimizers(self):
        parameters = self.model.parameters()
        ss_parameters = set(self.model.soundstream.parameters())
        remain_parameters = [p for p in parameters if p not in ss_parameters]
        optimizer, scheduler = create_optimizer(
            self.hp.training,
            remain_parameters,
            self.hp.trainer.total_steps
        )
        return [optimizer], [scheduler]

    def _training_loop(self, batch, batch_idx):
        f0_in = None
        if self.model.f0 is not None:
            f0_in = batch['f0']
        if self.token_type == 'soundstream':
            output = self.model(batch['mel'], f0=f0_in)
        else:
            if self.deduplicate:
                output = self.model(batch['dedup_tokens'], f0=f0_in)
            else:
                output = self.model(batch['tokens'], f0=f0_in)
        kld = masked_ce_loss(output['logits'],
                             output['labels'])
        loss = kld
        f0_loss = None
        if self.model.f0 is not None:
            f0_loss = masked_l1_loss(output['f0'], batch['f0'])
            loss += f0_loss * 0.5
        self.manual_backward(loss)
        return kld, f0_loss, output['logits']

    def training_step(self, batch, batch_idx):
        opt, sch = self.optimizers(), self.lr_schedulers()
        kld, f0_loss, logits = self._training_loop(batch, batch_idx)
        if (batch_idx + 1) % self.gradient_update_step == 0:
            if self.hp.training.has("gradient_clip_val"):
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.hp.training.gradient_clip_val,
                    gradient_clip_algorithm="norm")
            opt.step()
            opt.zero_grad(True)
            # Logging
            normalizing_length = logits.length.sum()
            self.log("train/kld", kld / normalizing_length,
                     prog_bar=True)
            current_lr = sch.get_last_lr()
            if not isinstance(current_lr, float):
                current_lr = current_lr[0]
            self.log("train/lr", current_lr)
            if f0_loss is not None:
                self.log("train/f0_loss", f0_loss / normalizing_length,
                         prog_bar=True)
            sch.step()

    def validation_step(self, batch, batch_idx):
        f0_in = None
        if self.model.f0 is not None:
            f0_in = batch['f0']
        if self.token_type == 'soundstream':
            output = self.model(batch['mel'], f0=f0_in)
        else:
            if self.deduplicate:
                output = self.model(batch['dedup_tokens'], f0=f0_in)
            else:
                output = self.model(batch['tokens'], f0=f0_in)
        kld = masked_ce_loss(output['logits'],
                             output['labels'])
        if self.model.f0 is not None:
            f0_loss = masked_l1_loss(output['f0'], batch['f0'])
        if self.sampled < self.hp.logging.num_samples:
            if self.token_type == 'soundstream':
                prior = batch['mel'].value
                prior_length = int(self.hp.logging.sample_prior_length *
                                   self.val_mel_sample_rate)
                length = int(self.hp.logging.sample_length *
                             self.val_mel_sample_rate *
                             self.model.sample_ratio)
            else:  # HuBERT
                if self.deduplicate:
                    prior = batch['dedup_tokens']
                    min_length = torch.min(prior.length)
                    prior = prior.value[:, :min_length]
                    prior_length = self.hp.logging.sample_prior_tokens
                    length = self.hp.logging.sample_tokens
                else:
                    prior = batch['tokens'].value
                    prior_length = int(self.hp.logging.sample_prior_length *
                                       self.val_token_sample_rate)
                    length = int(self.hp.logging.sample_length *
                                 self.val_token_sample_rate)
            prior = prior[:, :prior_length]
            if self.model.f0 is not None:
                prior_f0 = batch['f0'].value[:, :prior_length]
                prior = torch.cat([prior[..., None], prior_f0[..., None]], -1)
            args = [length, prior]
            kwargs = {
                'temperature': self.hp.logging.temperature,
                'return_attn': self.hp.logging.plot_attn
            }
            if self.model.soundstream.model.hp.has("spkr"):
                kwargs['spkr'] = batch['cropped_mel']
            samples = self.sampler(*args, **kwargs)
            re_vocoded = self.model.soundstream.vocoder.decode(batch['mel'])
            sampled_audio = samples["output"]
            for audio, sampled_aud, re_voc in zip(
                batch['audio'].tolist(),
                sampled_audio.tolist(),
                re_vocoded.tolist()
            ):
                if self.sampled < self.hp.logging.num_samples:
                    sw = self.logger.experiment
                    sw.add_audio(f're_vocoded/{self.sampled}',
                                 re_voc.float(),
                                 self.global_step,
                                 self.hp.data.train.sample_rate)
                    sw.add_audio(f'original/{self.sampled}',
                                 audio.float(),
                                 self.global_step,
                                 self.hp.data.val.sample_rate)
                    sw.add_audio(f'samples/{self.sampled}',
                                 sampled_aud.float(),
                                 self.global_step,
                                 self.hp.data.val.sample_rate)
                    self.sampled += 1
        normalizing_length = output['logits'].length.sum()
        self.log("val/kld", kld / normalizing_length,
                 on_epoch=True, logger=True, sync_dist=True,
                 batch_size=len(batch['audio']))
        if self.model.f0 is not None:
            self.log("val/f0_loss", f0_loss / normalizing_length,
                     on_epoch=True, logger=True, sync_dist=True,
                     batch_size=len(batch['audio']))

    def on_validation_start(self):
        self.sampled = 0

    def on_validation_end(self):
        self.sampled = 0

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)
        self.hp.save(os.path.join(self.logger.log_dir, 'hp.yaml'))
