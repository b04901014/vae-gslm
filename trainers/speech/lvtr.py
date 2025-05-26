from models.speech.lvtr import LVTR
from training_lib.trainer import BaseTrainer
from training_lib.optimizer import create_optimizer
from data.dataset import MelSpecDataset, DiscreteTokenDataset
from training_lib.losses import masked_loss
from hparams.hp import Hparams
from models.vocoder.vocoder import HiFiGAN
from .sampler import ARTRSampler
import torch
import os


class LVTRTrainer(BaseTrainer):
    def __init__(self, hp: Hparams) -> None:
        super().__init__(hp)
        self.hp = hp
        hp.check_arg_in_hparams("vocoder")
        hp.vocoder.check_arg_in_hparams("path")
        self.mel_rescale = None
        self.rec_loss_scale = self.hp.training.get("rec_loss_scale", 1.0)
        self.fixed_beta = self.hp.training.get("fixed_beta", None)

        if hp.training.has("mel_rescale"):
            hp.training.mel_rescale.check_arg_in_hparams("mean", "std")
            self.mel_rescale = hp.training.mel_rescale
        vocoder = HiFiGAN.from_pretrained(hp.vocoder.path,
                                          hp_rescale=self.mel_rescale)
        self.model = LVTR(hp.model, input_dim=vocoder.hp.n_mels)
        hp.check_arg_in_hparams("logging")
        hp.logging.check_arg_in_hparams("num_samples",
                                        "temperature",
                                        "sample_length",
                                        "sample_prior_length",
                                        "plot_attn")
        self.apply(self.init_weights)
        self.sampler = ARTRSampler(self.model)
        self.vocoder = vocoder
        for param in self.vocoder.parameters():
            param.requires_grad = False
        self.automatic_optimization = False
        self.zero_kld = hp.training.scheduler.get("zero_kld", 0)
        self.warmup_kld = hp.training.scheduler.get("warmup_kld", 0)
        self.use_tokens = self.model.use_tokens
        self.token_kld_weight = hp.training.get('token_kld_weight', 1.0)
        if self.use_tokens:
            hp.check_arg_in_hparams("hubert")
            hp.hubert.check_arg_in_hparams("sample_rate")
            self.hp_hubert = Hparams(deduplicate=False,
                                     sample_rate=hp.hubert.sample_rate)

    def train_dataloader(self):
        if self.use_tokens:
            train_dataset = DiscreteTokenDataset(self.hp.data.train,
                                                 self.vocoder.hp,
                                                 self.hp_hubert,
                                                 self.mel_rescale,
                                                 name="train dataset")
        else:
            train_dataset = MelSpecDataset(self.hp.data.train,
                                           self.vocoder.hp,
                                           self.mel_rescale,
                                           name="train dataset")
        return self.get_dataloader(self.hp.data.train, train_dataset)

    def val_dataloader(self):
        if self.use_tokens:
            val_dataset = DiscreteTokenDataset(self.hp.data.val,
                                               self.vocoder.hp,
                                               self.hp_hubert,
                                               self.mel_rescale,
                                               name="train dataset")
        else:
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
        kld_weight = 1.0
        if self.warmup_kld > 0 and ((self.global_step + 1) > self.zero_kld and
                                    (self.global_step + 1) <= self.warmup_kld):
            multiplier = (self.global_step - self.zero_kld) / self.warmup_kld
            kld_weight = multiplier
        if self.zero_kld > 0 and self.global_step <= self.zero_kld:
            kld_weight = 0.0
        kwargs = dict()
        if self.model.utterance_encoder is not None:
            kwargs['utterance'] = batch['cropped_mel_utt']
        if 'cropped_mel' in batch:
            kwargs['diff_input'] = batch['cropped_mel']
        model_input = batch['mel']
        if self.use_tokens:
            model_input = batch['tokens'].expand().cat(batch['mel'])
        output = self.model(model_input, **kwargs)
        entropy = output['log_q']
        log_p = output['log_p']
        kld = masked_loss(entropy,
                          log_p,
                          fn=lambda x, y: (x - y))
        rec_loss = output['decoder_output']
        rec_loss_opt = rec_loss * self.rec_loss_scale
        loss = rec_loss_opt + kld * kld_weight
        if self.use_tokens:
            token_kld = output['ce_loss']
            loss += token_kld * self.token_kld_weight * kld_weight
        self.manual_backward(loss)
        _output = {
            'kld': kld,
            'rec_loss': rec_loss,
            'log_p': -output['log_p'].mean(),
            'length': output['log_p'].length.sum(),
            'kld_weight': kld_weight
        }
        _output['logstd'] = output['logstd']
        _output['q_logstd'] = output['q_logstd']
        _output['log_q'] = -output['log_q'].mean()
        _output['q_mean_abs'] = output['q_mean_abs']
        if self.use_tokens:
            _output['token_kld'] = token_kld
        return _output

    def training_step(self, batch, batch_idx):
        opt, sch = self.optimizers(), self.lr_schedulers()
        output = self._training_loop(batch, batch_idx)
        if (batch_idx + 1) % self.gradient_update_step == 0:
            if self.hp.training.has("gradient_clip_val"):
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.hp.training.gradient_clip_val,
                    gradient_clip_algorithm="norm")
            opt.step()
            opt.zero_grad(True)
            # Logging
            normalizing_length = output['length']
            self.log("train/kld", output['kld'] / normalizing_length,
                     prog_bar=True)
            self.log("train/rec_loss", output['rec_loss'] / normalizing_length,
                     prog_bar=True)
            self.log("train/kld_weight", output['kld_weight'])
            self.log("train/z_given_logstd", output['logstd'],
                     prog_bar=True)
            self.log("train/q_logstd", output['q_logstd'],
                     prog_bar=True)
            self.log("train/q_entropy", output['log_q'])
            self.log("train/q_mean_abs", output['q_mean_abs'])
            self.log("train/cross_entropy", output['log_p'])
            current_lr = sch.get_last_lr()
            if not isinstance(current_lr, float):
                current_lr = current_lr[0]
            self.log("train/lr", current_lr)
            if self.use_tokens:
                self.log('train/token_kld',
                         output['token_kld'] / normalizing_length,
                         prog_bar=True)
            sch.step()

    def validation_step(self, batch, batch_idx):
        kwargs = dict()
        if self.model.utterance_encoder is not None:
            kwargs['utterance'] = batch['cropped_mel_utt']
        if 'cropped_mel' in batch:
            kwargs['diff_input'] = batch['cropped_mel']
        model_input = batch['mel']
        if self.use_tokens:
            model_input = batch['tokens'].expand().cat(batch['mel'])
        output = self.model(model_input, **kwargs)
        log_p = output['log_p']
        kld = masked_loss(output['log_q'], log_p,
                          fn=lambda x, y: (x - y))
        rec_loss = output['decoder_output']
        if self.use_tokens:
            token_kld = output['ce_loss']
        if self.sampled < self.hp.logging.num_samples:
            diff_cond = output['sample_q']
            if self.use_tokens:
                diff_cond = batch['tokens'].expand().cat(diff_cond)
            if self.model.utterance_encoder is not None:
                rec_audio = self.model.decode(diff_cond,
                                              u_c=output['u_c'])
            else:
                rec_audio = self.model.decode(diff_cond)
            prior = batch['mel'].value
            prior_length = int(self.hp.logging.sample_prior_length *
                               self.val_mel_sample_rate)
            prior = prior[:, :prior_length]
            if self.use_tokens:
                prior = torch.cat([(batch['tokens'].value
                                   [:, :prior_length, None]),
                                   prior],
                                  -1)
            length = int(self.hp.logging.sample_length *
                         self.val_mel_sample_rate *
                         self.model.sample_ratio)
            samples = self.sampler(
                length, prior,
                temperature=self.hp.logging.temperature,
                return_attn=self.hp.logging.plot_attn
            )
            rec_audio = self.vocoder.decode(rec_audio)
            re_vocoded = self.vocoder.decode(batch['mel'])
            sampled_audio = self.vocoder.decode(samples["output"])
            j = 0
            for audio, rec_aud, sampled_aud, re_voc in zip(
                batch['audio'].tolist(),
                rec_audio.tolist(),
                sampled_audio.tolist(),
                re_vocoded.tolist()
            ):
                if self.sampled < self.hp.logging.num_samples:
                    sw = self.logger.experiment
                    sw.add_audio(f're_vocoded/{self.sampled}',
                                 re_voc.float(),
                                 self.global_step,
                                 self.hp.data.train.sample_rate)
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
                    self.sampled += 1
                    j += 1
        normalizing_length = output['log_p'].length.sum()
        self.log("val/kld", kld / normalizing_length,
                 on_epoch=True, logger=True, sync_dist=True,
                 batch_size=len(batch['audio']))
        self.log("val/rec_loss", rec_loss / normalizing_length,
                 on_epoch=True, logger=True, sync_dist=True,
                 batch_size=len(batch['audio']))
        if self.use_tokens:
            self.log("val/token_kld", token_kld / normalizing_length,
                     on_epoch=True, logger=True, sync_dist=True,
                     batch_size=len(batch['audio']))

    def on_validation_start(self):
        self.sampled = 0

    def on_validation_end(self):
        self.sampled = 0

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)
        self.hp.save(os.path.join(self.logger.log_dir, 'hp.yaml'))
