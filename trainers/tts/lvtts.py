from models.tts.lvtr import LVTTS
from training_lib.trainer import BaseTrainer
from training_lib.optimizer import create_optimizer
from data.dataset import MelSpecDataset
from training_lib.losses import masked_loss, masked_l1_loss, InfoNCE, eos_loss
from hparams.hp import Hparams
from models.vocoder.vocoder import HiFiGAN
from utils.plots import plot_attn
from .sampler import ARTRTTSSampler
import torch
from pathlib import Path
import os


class LVTTSTrainer(BaseTrainer):
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
        self.train_dataset = MelSpecDataset(self.hp.data.train,
                                            vocoder.hp,
                                            self.mel_rescale,
                                            name="train dataset")
        self.model = LVTTS(hp.model,
                           symbols=self.train_dataset.symbols,
                           input_dim=vocoder.hp.n_mels)
        hp.check_arg_in_hparams("logging")
        hp.logging.check_arg_in_hparams("num_samples",
                                        "temperature",
                                        "max_sample_length",
                                        "min_sample_length",
                                        "plot_attn")
        self.infoNCE_weight = 1.0
        if hp.training.has("infoNCE"):
            self.model.infoNCE = InfoNCE(
                hp.training.infoNCE,
                dim1=hp.model.latent_dim,
                dim2=hp.model.latent_dim)
            self.infoNCE_weight = hp.training.infoNCE.get("weight", 1.0)
        self.apply(self.init_weights)
        self.vocoder = vocoder
        self.sampler = ARTRTTSSampler(self.model)
        for param in self.vocoder.parameters():
            param.requires_grad = False
        self.automatic_optimization = False
        self.run_infoNCE = hp.training.has("infoNCE")
        self.run_diffusion = hp.model.decoder.has('diffusion')
        self.rec_loss_scale = self.hp.training.get("rec_loss_scale", 1.0)

    def train_dataloader(self):
        return self.get_dataloader(self.hp.data.train, self.train_dataset)

    def val_dataloader(self):
        val_dataset = MelSpecDataset(self.hp.data.val,
                                     self.vocoder.hp,
                                     self.mel_rescale,
                                     name="validation dataset")
        val_dataset.symbols = self.train_dataset.symbols
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
        output = self.model(batch['mel'], batch['text'], batch['cropped_mel'])
        kld = masked_loss(output['log_q'],
                          output['log_p'],
                          fn=lambda x, y: (x - y))
        if self.run_infoNCE:
            infoNCE_loss = self.model.infoNCE(
                output['q_z'].sample,
                output['cnn_z'].detach()
            )
        else:
            infoNCE_loss = 0.0
        if self.run_diffusion:
            rec_loss = output['decoder_output']
        else:
            rec_loss = masked_l1_loss(batch['mel'], output['decoder_output'])
        _eos_loss = eos_loss(output["eos"])
        rec_loss_opt = rec_loss * self.rec_loss_scale
        loss = (rec_loss_opt + kld + infoNCE_loss * self.infoNCE_weight +
                _eos_loss)
        self.manual_backward(loss)
        return (kld,
                rec_loss,
                output['logstd'],
                output['q_logstd'],
                infoNCE_loss,
                -output['log_q'].mean(),
                -output['log_p'].mean(),
                _eos_loss)

    def training_step(self, batch, batch_idx):
        opt, sch = self.optimizers(), self.lr_schedulers()
        (kld,
         rec_loss,
         logstd,
         q_lostd,
         infoNCE_loss,
         q_entropy,
         cross_entropy,
         _eos_loss) = self._training_loop(batch, batch_idx)
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
        self.log("train/kld", kld,
                 prog_bar=True)
        self.log("train/rec_loss", rec_loss,
                 prog_bar=True)
        self.log("train/z_given_logstd", logstd,
                 prog_bar=True)
        self.log("train/q_logstd", q_lostd,
                 prog_bar=True)
        self.log("train/q_entropy", q_entropy)
        self.log("train/cross_entropy", cross_entropy)
        self.log("train/infoNCE", infoNCE_loss, prog_bar=True)
        self.log("train/eos", _eos_loss)

    def validation_step(self, batch, batch_idx):
        output = self.model(batch['mel'], batch['text'], batch['cropped_mel'])
        kld = masked_loss(output['log_q'], output['log_p'],
                          fn=lambda x, y: (x - y))
        if self.hp.model.decoder.has('diffusion'):
            rec_loss = output['decoder_output']
        else:
            rec_loss = masked_l1_loss(batch['mel'], output['decoder_output'])
        if self.hp.training.has("infoNCE"):
            infoNCE_loss = self.model.infoNCE(
                output['q_z'].sample,
                output['cnn_z']
            )
        else:
            infoNCE_loss = 0.0
        _eos_loss = eos_loss(output["eos"])
        if self.sampled < self.hp.logging.num_samples:
            if self.hp.model.decoder.has('diffusion'):
                rec_audio = self.model.decode(output['sample_q'],
                                              output['condition'])
            else:
                rec_audio = output['decoder_output'].apply_mask()
            rec_audio = self.vocoder.decode(rec_audio)
            samples = self.sampler(
                batch['text'],
                batch['cropped_mel'],
                int(self.hp.logging.max_sample_length *
                    self.val_mel_sample_rate *
                    self.model.sample_ratio),
                int(self.hp.logging.min_sample_length *
                    self.val_mel_sample_rate *
                    self.model.sample_ratio),
                temperature=self.hp.logging.temperature,
                eos_threshold=self.hp.logging.get("eos_threshold", 0.5),
                return_attn=self.hp.logging.plot_attn
            )
            sampled_audio = self.vocoder.decode(samples["output"])
            for ijk, (audio, rec_aud, sampled_aud, text) in enumerate(zip(
                batch['audio'].tolist(),
                rec_audio.tolist(),
                sampled_audio.tolist(),
                batch['text_written_form']
            )):
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
                    sw.add_audio(f'samples/{self.sampled}',
                                 sampled_aud.float(),
                                 self.global_step,
                                 self.hp.data.val.sample_rate)
                    sw.add_text(f'text/{self.sampled}',
                                text,
                                self.global_step)
                    if self.hp.logging.plot_attn:
                        plot_attn(sw, self.global_step,
                                  samples['self_attn'][ijk],
                                  f'self_attn/{self.sampled}', (10, 10))
                        plot_attn(sw, self.global_step,
                                  samples['text_self_attn'][ijk],
                                  f'text_self_attn/{self.sampled}', (10, 10))
                        plot_attn(sw, self.global_step,
                                  samples['cross_attn'][ijk],
                                  f'cross_attn/{self.sampled}', (10, 10))
                    self.sampled += 1
        self.log("val/kld", kld,
                 on_epoch=True, logger=True, sync_dist=True,
                 batch_size=len(batch['audio']))
        self.log("val/rec_loss", rec_loss,
                 on_epoch=True, logger=True, sync_dist=True,
                 batch_size=len(batch['audio']))
        self.log("val/infoNCE", infoNCE_loss,
                 on_epoch=True, logger=True, sync_dist=True,
                 batch_size=len(batch['audio']))
        self.log("val/eos", _eos_loss,
                 on_epoch=True, logger=True, sync_dist=True,
                 batch_size=len(batch['audio']))

    def on_validation_start(self):
        self.sampled = 0

    def on_validation_end(self):
        self.sampled = 0

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)
        self.train_dataset.symbols.save(
            str(Path(path).parent / "symbols.json"))
        self.hp.save(os.path.join(self.logger.log_dir, 'hp.yaml'))
