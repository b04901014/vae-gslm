from hparams.hp import Hparams
from .likelihood import LikelihoodEstimator
from modules.diffusion.ddpm import GaussianDiffusion1D
from utils import crepe, mcd
import torchcrepe
import torch
from transformers import Wav2Vec2Processor, HubertForCTC
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import jiwer


class ReconstructionEvaluator(LikelihoodEstimator):
    def __init__(self, hp: Hparams) -> None:
        super().__init__(hp)
        hp.check_arg_in_hparams("tasks")
        if 'f0mse' in hp.tasks:
            torchcrepe.load.model('cpu', 'tiny')
            self.f0_predictor = torchcrepe.infer.model
        if 'wer' in hp.tasks:
            hp.check_arg_in_hparams('asr_model')
            if hp.asr_model.split('/')[0] == 'facebook':
                self.asr_processor = Wav2Vec2Processor.from_pretrained(
                    hp.asr_model)
                self.asr_model = HubertForCTC.from_pretrained(hp.asr_model)
            else:
                self.asr_processor = WhisperProcessor.from_pretrained(
                    hp.asr_model)
                self.asr_model = (WhisperForConditionalGeneration.
                                  from_pretrained(hp.asr_model))
        if self.hp.model.identifier == "models.speech.discrete.DiscreteAR":
            self.vocoder = self.model.soundstream.vocoder
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

    def on_test_start(self):
        self.scores = {k: [] for k in self.hp.tasks}
        if 'wer' in self.hp.tasks:
            self.gt_transcripts, self.re_transcripts = [], []

    def on_test_end(self):
        if 'wer' in self.hp.tasks:
            transformation = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemoveWhiteSpace(replace_by_space=True),
                jiwer.RemoveMultipleSpaces(),
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemovePunctuation(),
                jiwer.Strip(),
                jiwer.ReduceToListOfListOfChars()
            ])
            self.scores['gt_wer'] = jiwer.cer(
                self.transcripts,
                self.gt_transcripts,
                reference_transform=transformation,
                hypothesis_transform=transformation
            )
            self.scores['re_wer'] = jiwer.cer(
                self.transcripts,
                self.re_transcripts,
                reference_transform=transformation,
                hypothesis_transform=transformation
            )

    def test_step(self, batch, batch_idx):
        gold = self.vocoder.decode(batch['mel'])
        if self.hp.model.identifier == "models.speech.discrete.DiscreteAR":
            f0 = batch.get('f0', None)
            rec = self.model.soundstream.decode(batch[self.input_key],
                                                spkr=batch['mel'],
                                                f0=f0)
        else:
            model_input = batch[self.input_key]
            if self.use_tokens:
                model_input = batch['tokens'].expand().cat(batch['mel'])
            utt = self.model.encode_utterance(model_input)
            rec = self.model.encode(model_input,
                                    temperature=0.0)  # TODO: Add Beta
            rec = self.model.decode(rec,
                                    u_c=utt)
            rec = self.vocoder.decode(rec)
        # Calculate f0 mse
        for _gold, _rec in zip(gold.tolist(), rec.tolist()):
            # Use masking from the generatred to ensure consistency
            # This is a workaround for HuBERT token length sightly mismatch
            _gold = _gold[:_rec.size(0)]
            if 'f0mse' in self.scores:
                f0_g, period = crepe.predict(_gold[None],
                                             self.vocoder.hp.sample_rate,
                                             self.f0_predictor, _gold.device,
                                             batch_size=2048,
                                             return_periodicity=True)
                f0_r = crepe.predict(_rec[None], self.vocoder.hp.sample_rate,
                                     self.f0_predictor, _rec.device,
                                     batch_size=2048)
                period = torchcrepe.filter.median(period, 3)
                period = torchcrepe.threshold.Silence(-60.)(
                    period, _gold[None], self.vocoder.hp.sample_rate)
                mse = (f0_g - f0_r).pow(2)
                mse = mse[period > 0.21].mean()
                self.scores['f0mse'].append(mse.item())
            if 'mcd' in self.scores:
                mcd_score = mcd.mcd(_gold.cpu().numpy(), _rec.cpu().numpy())
                self.scores['mcd'].append(mcd_score)
            if 'wer' in self.scores:
                # torchaudio.save('rrrrrr.wav', _rec.cpu()[None].float(), 16000)
                if self.hp.asr_model.split('/')[0] == 'facebook':
                    gt_transcript = self.asr_processor(_gold,
                                                       return_tensors="pt",
                                                       sampling_rate=16000)
                    gt_transcript = self.asr_model(
                        gt_transcript.input_values.to(_gold.dtype).cuda()
                    ).logits
                    gt_transcript = torch.argmax(gt_transcript, dim=-1)
                    gt_transcript = self.asr_processor.decode(gt_transcript[0])
                    re_transcript = self.asr_processor(_rec,
                                                       return_tensors="pt",
                                                       sampling_rate=16000)
                    re_transcript = self.asr_model(
                        re_transcript.input_values.to(_gold.dtype).cuda()
                    ).logits
                    re_transcript = torch.argmax(re_transcript, dim=-1)
                    re_transcript = self.asr_processor.decode(re_transcript[0])
                else:
                    gt_transcript = self.asr_processor(_gold.cpu().numpy(),
                                                       return_tensors="pt",
                                                       sampling_rate=16000)
                    gt_transcript = self.asr_model.generate(
                        gt_transcript.input_features.cuda(), language='en'
                    )
                    gt_transcript = self.asr_processor.batch_decode(
                        gt_transcript,
                        skip_special_tokens=True)[0]
                    re_transcript = self.asr_processor(_rec.cpu().numpy(),
                                                       return_tensors="pt",
                                                       sampling_rate=16000)
                    re_transcript = self.asr_model.generate(
                        re_transcript.input_features.cuda(), language='en'
                    )
                    re_transcript = self.asr_processor.batch_decode(
                        re_transcript,
                        skip_special_tokens=True)[0]
                self.gt_transcripts.append(gt_transcript)
                self.re_transcripts.append(re_transcript)
