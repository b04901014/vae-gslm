from models.speech.soundstream import SoundStream
from training_lib.trainer import BaseTrainer
from training_lib.optimizer import create_optimizer
from data.dataset import MelSpecDataset
from training_lib.losses import masked_l1_loss
from hparams.hp import Hparams
from models.vocoder.vocoder import HiFiGAN
import torch
import os


class SoundStreamTrainer(BaseTrainer):
    def __init__(self, hp: Hparams) -> None:
        super().__init__(hp)
        self.hp = hp
        hp.check_arg_in_hparams("vocoder")
        hp.vocoder.check_arg_in_hparams("path")
        self.mel_rescale = None
        if hp.training.has("mel_rescale"):
            hp.training.mel_rescale.check_arg_in_hparams("mean", "std")
            self.mel_rescale = hp.training.mel_rescale
        vocoder = HiFiGAN.from_pretrained(hp.vocoder.path,
                                          hp_rescale=self.mel_rescale)
        for param in vocoder.parameters():
            param.requires_grad = False
        self.model = SoundStream(hp.model, input_dim=vocoder.hp.n_mels)
        hp.check_arg_in_hparams("logging")
        hp.logging.check_arg_in_hparams("num_samples")
        self.apply(self.init_weights)
        self.vocoder = vocoder
        self.automatic_optimization = False

    def train_dataloader(self):
        train_dataset = MelSpecDataset(self.hp.data.train,
                                       self.vocoder.hp,
                                       self.mel_rescale,
                                       name="train dataset")
        return self.get_dataloader(self.hp.data.train, train_dataset)

    def val_dataloader(self):
        val_dataset = MelSpecDataset(self.hp.data.val,
                                     self.vocoder.hp,
                                     self.mel_rescale,
                                     name="validation dataset")
        self.val_mel_sample_rate = val_dataset.melspec.sample_rate
        return self.get_dataloader(self.hp.data.val, val_dataset)

    def configure_optimizers(self):
        optimizer, scheduler = create_optimizer(
            self.hp.training,
            self.model.parameters(),
            self.hp.trainer.total_steps
        )
        return [optimizer], [scheduler]

    def _training_loop(self, batch, batch_idx):
        output = self.model(batch['mel'])
        rec_loss = masked_l1_loss(batch['mel'],
                                  output['reconstruction'],
                                  time_reduction=True,
                                  batch_reduction=True)
        aux_loss = output['aux_loss']
        loss = rec_loss
        if aux_loss is not None:
            aux_loss = aux_loss.mean()
            loss += aux_loss
        self.manual_backward(loss)
        return rec_loss, aux_loss

    def training_step(self, batch, batch_idx):
        opt, sch = self.optimizers(), self.lr_schedulers()
        rec_loss, aux_loss = self._training_loop(batch, batch_idx)
        if self.hp.training.has("gradient_clip_val"):
            self.clip_gradients(
                opt,
                gradient_clip_val=self.hp.training.gradient_clip_val,
                gradient_clip_algorithm="norm")
        if batch_idx % self.gradient_update_step == 0:
            opt.step()
            sch.step()
            opt.zero_grad(True)
        # Logging
        self.log("train/rec_loss", rec_loss,
                 on_step=True, prog_bar=True,
                 batch_size=len(batch['audio']))
        if aux_loss is not None:
            self.log("train/aux_loss", aux_loss,
                     on_step=True, prog_bar=True,
                     batch_size=len(batch['audio']))

    def validation_step(self, batch, batch_idx):
        output = self.model(batch['mel'])
        rec_loss = masked_l1_loss(batch['mel'],
                                  output['reconstruction'],
                                  time_reduction=True,
                                  batch_reduction=True)
        aux_loss = output['aux_loss']
        rec_audio = self.vocoder.decode(output['reconstruction'])
        if self.sampled < self.hp.logging.num_samples:
            for audio, rec_aud in zip(
                batch['audio'].tolist(),
                rec_audio.tolist()
            ):
                if self.sampled < self.hp.logging.num_samples:
                    sw = self.logger.experiment
                    sw.add_audio(f'reconstruct/{self.sampled}',
                                 rec_aud.float(),
                                 self.global_step,
                                 self.hp.data.train.sample_rate)
                    sw.add_audio(f'original/{self.sampled}',
                                 audio.float(),
                                 self.global_step,
                                 self.hp.data.val.sample_rate)
                    self.sampled += 1
        self.log("val/rec_loss", rec_loss,
                 on_epoch=True, logger=True, sync_dist=True,
                 batch_size=len(batch['audio']))
        if aux_loss is not None:
            aux_loss = aux_loss.mean()
            self.log("val/aux_loss", aux_loss,
                     on_epoch=True, logger=True, sync_dist=True,
                     batch_size=len(batch['audio']))

    def on_validation_start(self):
        self.sampled = 0

    def on_validation_end(self):
        self.sampled = 0

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)
        self.hp.save(os.path.join(self.logger.log_dir, 'hp.yaml'))
