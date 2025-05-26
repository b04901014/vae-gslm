from models.vocoder.hfgan import (Generator, MultiScaleDiscriminator,
                                  MultiResolutionDiscriminator,
                                  MultiPeriodDiscriminator)
from models.vocoder.hfgan import (feature_loss, discriminator_loss,
                                  generator_loss)
from training_lib.trainer import BaseTrainer
from training_lib.optimizer import create_optimizer
from data.features import MelSpecFeatureProcessor
from data.dataset import StandardDataset
from itertools import chain
from training_lib.losses import masked_l1_loss
from utils.plots import plot_spectrogram
from hparams.hp import Hparams
import torch
import os


class HiFiGANTrainer(BaseTrainer):
    def __init__(self, hp: Hparams) -> None:
        super().__init__(hp)
        self.hp = hp
        hp.model.check_arg_in_hparams("mpd", "generator")
        hp.training.check_arg_in_hparams("generator", "discriminator",
                                         "mel_loss_weight")
        hp.check_arg_in_hparams("logging", "feature")
        hp.logging.check_arg_in_hparams("num_samples")
        self.mpd = MultiPeriodDiscriminator(hp.model.mpd)
        if hp.model.get("msd", False):
            self.msrd = MultiScaleDiscriminator(hp.model.msd)
        else:
            hp.model.check_arg_in_hparams("mrd")
            self.msrd = MultiResolutionDiscriminator(hp.model.mrd)
        self.generator = Generator(hp.model.generator)
        self.feature_processor = MelSpecFeatureProcessor(hp.feature)
        self.apply(self.init_weights)

    def train_dataloader(self):
        train_dataset = StandardDataset(self.hp.data.train,
                                        name="train dataset")
        return self.get_dataloader(self.hp.data.train, train_dataset)

    def val_dataloader(self):
        val_dataset = StandardDataset(self.hp.data.val,
                                      name="validation dataset")
        return self.get_dataloader(self.hp.data.val, val_dataset)

    @property
    def automatic_optimization(self):
        return False

    @property
    def joint_global_step(self):
        return self.global_step // 2

    def configure_optimizers(self):
        opt_g, scheduler_g = create_optimizer(self.hp.training.generator,
                                              self.generator.parameters(),
                                              self.hp.trainer.total_steps // 2)
        opt_d, scheduler_d = create_optimizer(self.hp.training.discriminator,
                                              chain(self.mpd.parameters(),
                                                    self.msrd.parameters()),
                                              self.hp.trainer.total_steps // 2)
        return [opt_g, opt_d], [scheduler_g, scheduler_d]

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()
        y_g_mel = self.feature_processor.encode(batch['audio'])
        y_g_hat = self.generator(y_g_mel)
        y_g_hat_mel = self.feature_processor.encode(y_g_hat)
        y_g_hat = y_g_hat.value
        y = batch['audio'].value

        # Discriminator
        opt_d.zero_grad()
        y_df_hat_r, fmap_f_r = self.mpd(y.unsqueeze(1))
        fmap_f_r = [[x.detach() for x in f] for f in fmap_f_r]
        y_df_hat_g, _ = self.mpd(y_g_hat.detach().unsqueeze(1))
        loss_disc_f = discriminator_loss(y_df_hat_r, y_df_hat_g)
        y_ds_hat_r, fmap_s_r = self.msrd(y.unsqueeze(1))
        fmap_s_r = [[x.detach() for x in f] for f in fmap_s_r]
        y_ds_hat_g, _ = self.msrd(y_g_hat.detach().unsqueeze(1))
        loss_disc_s = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        loss_disc_all = loss_disc_s + loss_disc_f
        self.manual_backward(loss_disc_all)
        opt_d.step()
        sch_d.step()

        # Generator
        opt_g.zero_grad()
        loss_mel = masked_l1_loss(y_g_hat_mel, y_g_mel,
                                  time_reduction=True,
                                  batch_reduction=True)
        y_df_hat_g, fmap_f_g = self.mpd(y_g_hat.unsqueeze(1))
        y_ds_hat_g, fmap_s_g = self.msrd(y_g_hat.unsqueeze(1))
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = (loss_gen_s + loss_gen_f +
                        loss_fm_s + loss_fm_f +
                        loss_mel * self.hp.training.mel_loss_weight)
        self.manual_backward(loss_gen_all)
        opt_g.step()
        sch_g.step()

        # Logging
        self.log("train/mel", loss_mel, on_step=True, prog_bar=True,
                 batch_size=len(batch['audio']))
        self.log("train/G", loss_gen_s + loss_gen_f,
                 on_step=True, prog_bar=True,
                 batch_size=len(batch['audio']))
        self.log("train/feature", loss_fm_s + loss_fm_f,
                 on_step=True, prog_bar=True,
                 batch_size=len(batch['audio']))
        self.log("train/D", loss_disc_s + loss_disc_f,
                 on_step=True, prog_bar=True,
                 batch_size=len(batch['audio']))

    def validation_step(self, batch, batch_idx):
        y_g_mel = self.feature_processor.encode(batch['audio'])
        y_g_hat = self.generator(y_g_mel)
        y_g_hat_mel = self.feature_processor.encode(y_g_hat)
        loss_mel = masked_l1_loss(y_g_hat_mel, y_g_mel,
                                  time_reduction=True,
                                  batch_reduction=True)
        if self.sampled < self.hp.logging.num_samples:
            for mel, rec_mel, wav, rec_wav in zip(y_g_mel.tolist(),
                                                  y_g_hat_mel.tolist(),
                                                  batch['audio'].tolist(),
                                                  y_g_hat.tolist()):
                if self.sampled < self.hp.logging.num_samples:
                    sw = self.logger.experiment
                    sw.add_audio(f'reconstruct/{self.sampled}',
                                 rec_wav.float(),
                                 self.joint_global_step,
                                 self.hp.data.train.sample_rate)
                    sw.add_audio(f'original/{self.sampled}',
                                 wav.float(),
                                 self.joint_global_step,
                                 self.hp.data.val.sample_rate)
                    sw.add_figure(f'reconstruct-mel/{self.sampled}',
                                  plot_spectrogram(rec_mel.float().
                                                   cpu().numpy()),
                                  self.joint_global_step)
                    sw.add_figure(f'original-mel/{self.sampled}',
                                  plot_spectrogram(mel.float().cpu().numpy()),
                                  self.joint_global_step)
                    self.sampled += 1
        self.log("val/mel", loss_mel,
                 on_epoch=True, logger=True, sync_dist=True,
                 batch_size=len(batch['audio']))

    def on_validation_start(self):
        self.sampled = 0

    def on_validation_end(self):
        self.sampled = 0

    def save_checkpoint(self, path: str):
        torch.save(self.generator.state_dict(), path)
        self.hp.save(os.path.join(self.logger.log_dir, 'hp.yaml'))
