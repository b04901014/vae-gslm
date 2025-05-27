from hparams.hp import Hparams
import logging
from typing import Tuple, List, Optional, Set, Mapping, Any
from collections.abc import Iterable
from torch.utils import data
import os
import torchaudio
import torch
from utils.helpers import (random_crop_1d, truncate_1d,
                           pad_1d, pad_to_max_length)
from .symbols import Symbols
import math
from data.features import MelSpecFeatureProcessor
import numpy as np
from pathlib import Path

SAMPLE_RATE_POOL = [16000, 44100, 48000, 24000]


def load_dataset(metadata: str,
                 with_text: bool,
                 delimiter: str = ' ',
                 min_audio_length: Optional[bool] = None,
                 max_audio_length: Optional[bool] = None,
                 bits_per_second: Optional[int] = None,
                 wavdir: Optional[str] = '',
                 max_text_tokens: Optional[int] = 2 ** 63 - 1,
                 min_text_tokens: Optional[int] = 0,
                 with_tokens: bool = False,
                 max_token_length: Optional[int] = 2 ** 63 - 1,
                 min_token_length: Optional[int] = 0
                 ) -> Tuple[List[str], List[str], Set, List[float]]:
    """
    Load a dataset of the format specified in the README.md.
    Returns also the transcript and set of symbols
    if with_text is set, otherwise
    return empty lists for the last two arguments.

    Note that this method caches all text the in RAM, which may pose OOM issues
    for large dataset or small RAM machines.
    """
    filenames = []
    texts = []
    lengths = []
    tokens = []
    symbols = set()
    audio_lengths = 0
    logging.info(f"Loading Dataset from {metadata}...")
    logging.debug(f"Proceed with with_text={with_text}")
    if min_audio_length is not None or max_audio_length is not None:
        assert bits_per_second is not None
    with open(metadata, 'r', errors='ignore') as f:
        for line in f:
            added_length = False
            fn = line.strip()
            if not fn:
                continue
            if with_text:
                fn = fn.split('|')
                if len(fn) != 3:
                    raise ValueError("Number of delimiter `|` not correct"
                                     f", expected 3, got {len(fn)}")
            else:
                fn = fn.split('|', 1)
            if bits_per_second is not None:
                audio_length = (os.path.getsize(os.path.join(wavdir, fn[0])) /
                                float(bits_per_second))
                audio_lengths += audio_length
                if min_audio_length is not None:
                    if audio_length < min_audio_length:
                        continue
                if max_audio_length is not None:
                    if audio_length > max_audio_length:
                        continue
                lengths.append(audio_length)
                added_length = True
            filenames.append(fn[0])
            if with_text:
                sentence = fn[2].split(delimiter)
                ls = len(sentence)
                if ls > max_text_tokens or ls < min_text_tokens:
                    del filenames[-1]
                    if added_length:
                        del lengths[-1]
                    continue
                texts.append(sentence)
                symbols = symbols.union(set(sentence))
            if with_tokens:
                token = np.fromstring(fn[-1], dtype=np.int16, sep=' ')
                ls = len(token)
                if ls > max_token_length or ls < min_token_length:
                    del filenames[-1]
                    if added_length:
                        del lengths[-1]
                    continue
                token = torch.from_numpy(token)
                tokens.append(token)
    logging.info("Done loading dataset!")
    logging.debug(f"Number of lines presented in {metadata}: {len(filenames)}")
    logging.debug(f"All symbols presented in the transcripts: {symbols}")
    if min_audio_length is not None:
        logging.debug("Average length of audio clips: "
                      f"{audio_lengths / len(filenames)} sec")
    return filenames, texts, symbols, lengths, tokens


class StandardDataset(data.Dataset):
    def __init__(self,
                 hp: Hparams,
                 name: Optional[str] = None) -> None:
        hp.check_arg_in_hparams("with_text", "path",
                                "sample_rate", "wavdir")
        store_length = False
        if hp.has("sampler") and hp.sampler.type == "bucket":
            store_length = True
        if getattr(hp, 'segment_size', False):
            assert not hp.with_text, ("Not sure why need"
                                      " to random segment the audio"
                                      " with a TTS generation. "
                                      "Currently alignment mode"
                                      " not supported!")
        self.hp = hp
        if hp.with_text:
            hp.check_arg_in_hparams("delimiter")
        if hp.get("min_audio_length", False):
            hp.check_arg_in_hparams("bits_per_second")
        self.name = name if name is not None else "dataset"
        self.audios, self.texts, self.symbols, self.tokens = [], [], set(), []
        path, wavdir = hp.path, hp.wavdir
        bits_per_second = hp.get("bits_per_second", None)
        if isinstance(hp.path, str):
            assert isinstance(hp.wavdir, str)
            path, wavdir = [path], [wavdir]
        if not isinstance(bits_per_second, list):
            bits_per_second = [bits_per_second] * len(path)
        lengths = []
        for _path, _wavdir, _bps in zip(path, wavdir, bits_per_second):
            _audios, _texts, _symbols, _length, _tokens = load_dataset(
                _path,
                hp.with_text,
                hp.get("delimiter", ' '),
                hp.get("min_audio_length", None),
                hp.get("max_audio_length", None),
                _bps,
                _wavdir,
                hp.get("max_text_tokens", 1000000),
                hp.get("min_text_tokens", 0),
                hp.get("with_tokens", False),
                hp.get("max_token_length", 1000000),
                hp.get("min_token_length", 0)
            )
            _audios = [
                os.path.join(_wavdir, f)
                for f in _audios
            ]
            self.audios += _audios
            self.texts += _texts
            self.symbols = self.symbols.union(_symbols)
            self.tokens += _tokens
            lengths += _length
        if hp.with_text:
            self.symbols = Symbols(self.symbols, hp.delimiter)
        if store_length:
            logging.info(f"{name}: Use audios length"
                         " for better bucketing.")
            hp.check_arg_in_hparams("bits_per_second")
            self.lengths = lengths
            if hp.has('truncate'):
                self.lengths = [
                    min(length, hp.truncate)
                    for length in self.lengths
                ]
        logging.info(f'{name}: Total {len(self.audios)} examples')
        self.resamplers = {
            sr: torchaudio.transforms.Resample(sr, self.hp.sample_rate)
            for sr in SAMPLE_RATE_POOL if sr != self.hp.sample_rate
        }

    def __len__(self) -> int:
        return len(self.audios)

    def __getitem__(self, i):
        audio, sr = torchaudio.load(self.audios[i])
        # Currently only deal with single channel input
        audio = audio.mean(0)
        if self.hp.get("dither", False):
            audio = torchaudio.functional.dither(audio)
        if sr != self.hp.sample_rate:
            assert sr in self.resamplers, f"Sample rate: {sr} not supported."
            audio = self.resamplers[sr](audio)
        if self.hp.has('segment_size'):
            audio = random_crop_1d(audio, self.hp.sample_rate,
                                   self.hp.segment_size)
        if self.hp.has('pad'):
            assert not (self.hp.pad.has('length') and
                        self.hp.pad.has('multiple_of'))
            assert self.hp.pad.has('length') or self.hp.pad.has('multiple_of')
            padding_mode = getattr(self.hp.pad,
                                   "padding_mode",
                                   "constant")
            if self.hp.pad.has('length'):
                pad_length = self.hp.pad.length
            if self.hp.pad.has('multiple_of'):
                multiple = math.ceil(float(len(audio)) /
                                     float(self.hp.pad.multiple_of))
                pad_length = multiple * self.hp.pad.multiple_of
                pad_length = float(pad_length) / float(self.hp.sample_rate)
            audio = pad_1d(audio,
                           self.hp.sample_rate,
                           pad_length,
                           padding_mode)
        if self.hp.has('truncate'):
            audio = truncate_1d(audio,
                                self.hp.sample_rate,
                                self.hp.truncate)
        ret = {'audio': audio}
        if self.hp.with_text:
            encoded = self.symbols.encode(self.texts[i])
            if self.hp.has('pad_text'):
                assert self.hp.pad_text.has('length')
                encoded = pad_1d(encoded, 1, self.hp.pad_text.length)
            re_decoded = self.symbols.decode(encoded)
            ret['text'] = torch.LongTensor(encoded)
            ret['text_written_form'] = re_decoded
        return ret

    def get_post_pad_dict(self) -> Mapping[str, Any]:
        post_pad_dict = None
        if self.hp.has('post_pad'):
            if self.hp.post_pad.has('text'):
                assert self.hp.post_pad.text.has('length')
                if post_pad_dict is None:
                    post_pad_dict = dict()
                post_pad_dict['text'] = self.hp.post_pad.text.length
            if self.hp.post_pad.has('audio'):
                assert self.hp.post_pad.audio.has('length')
                if post_pad_dict is None:
                    post_pad_dict = dict()
                post_pad_dict['audio'] = int(self.hp.post_pad.audio.length *
                                             self.hp.sample_rate)
        return post_pad_dict

    def seqCollate(self, batch: Iterable[Mapping[str, Any]]
                   ) -> Mapping[str, Any]:
        post_pad_dict = self.get_post_pad_dict()
        output = pad_to_max_length(batch, post_pad_dict)
        return output


class MelSpecDataset(StandardDataset):
    def __init__(self,
                 hp: Hparams,
                 hp_mel: Hparams,
                 hp_rescale: Optional[Hparams] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(hp, name=name)
        self.melspec = MelSpecFeatureProcessor(hp_mel)
        if self.hp.has("random_crop_mel"):
            self.hp.random_crop_mel.check_arg_in_hparams("min_seg_sec",
                                                         "max_seg_sec")
        self.hp_rescale = hp_rescale
        self.preprocess_mels = self.hp.get("preprocess_mels", None)
        self.preprocess_mels_recursive_dir = self.hp.get(
            "preprocess_mels_recursive_dir", False)
        self.preprocess_f0 = self.hp.get("preprocess_f0", None)

    def __getitem__(self, i):
        if self.preprocess_mels is None:
            ret = super().__getitem__(i)
            mel = self.melspec.encode_single(ret['audio'])
        else:
            ret = dict()
            if self.hp.with_text:
                encoded = self.symbols.encode(self.texts[i])
                if self.hp.has('pad_text'):
                    assert self.hp.pad_text.has('length')
                    encoded = pad_1d(encoded, 1, self.hp.pad_text.length)
                re_decoded = self.symbols.decode(encoded)
                ret['text'] = torch.LongTensor(encoded)
                ret['text_written_form'] = re_decoded
            if self.preprocess_mels_recursive_dir:
                p = Path(self.audios[i])
                fname = p.parents[0] / Path(p.stem + '.npy')
                fname = str(fname.resolve())
                fname = fname[len(str(Path(self.hp.wavdir).resolve()))+1:]
                fmel = os.path.join(self.preprocess_mels, fname)
            else:
                fmel = os.path.join(self.preprocess_mels,
                                    Path(self.audios[i]).stem + '.npy')
            mel = torch.from_numpy(np.load(fmel).astype(np.float32))
        if self.preprocess_f0 is not None:
            if self.preprocess_mels_recursive_dir:
                p = Path(self.audios[i])
                fname = str(p.parents[0] / Path(p.stem + '.npy'))
                ff0 = os.path.join(self.preprocess_f0.path,
                                   fname[len(self.hp.wavdir):])
            else:
                ff0 = os.path.join(self.preprocess_f0.path,
                                   Path(self.audios[i]).stem + '.npy')
            f0 = torch.from_numpy(np.load(ff0).astype(np.float32))
            if self.preprocess_f0.get("log", True):
                f0 = torch.log(1 + f0)
            if self.preprocess_f0.get("normalize", True):
                mask = (f0 == 0)
                mean = f0[~mask].mean()
                f0 = torch.where(mask, 0, f0 - mean)
            f0 = f0[: mel.size(0)]
        if self.hp.has('segment_size'):
            mel, mel_s, mel_e = random_crop_1d(mel,
                                               self.melspec.sample_rate,
                                               self.hp.segment_size,
                                               return_start_end=True)
            if self.preprocess_f0 is not None:
                f0 = f0[mel_s: mel_e]
        if self.hp_rescale is not None:
            mel = (mel - self.hp_rescale.mean) / self.hp_rescale.std
        ret['mel'] = mel
        if self.preprocess_f0 is not None:
            ret['f0'] = f0
        if self.hp.has("random_crop_mel"):
            min_seg = self.hp.random_crop_mel.min_seg_sec
            max_seg = self.hp.random_crop_mel.max_seg_sec
            seg_len = np.random.rand() * (max_seg - min_seg) + min_seg
            cropped_mel = random_crop_1d(mel, self.melspec.sample_rate,
                                         seg_len)
            ret['cropped_mel'] = cropped_mel
        if self.hp.has("random_crop_mel_utt"):
            min_seg = self.hp.random_crop_mel_utt.min_seg_sec
            max_seg = self.hp.random_crop_mel_utt.max_seg_sec
            seg_len = np.random.rand() * (max_seg - min_seg) + min_seg
            cropped_mel = random_crop_1d(mel, self.melspec.sample_rate,
                                         seg_len)
            ret['cropped_mel_utt'] = cropped_mel
        return ret

    def get_post_pad_dict(self) -> Mapping[str, Any]:
        post_pad_dict = None
        if self.hp.has('post_pad'):
            if self.hp.post_pad.has('text'):
                assert self.hp.post_pad.text.has('length')
                if post_pad_dict is None:
                    post_pad_dict = dict()
                post_pad_dict['text'] = self.hp.post_pad.text.length
            if self.hp.post_pad.has('mel'):
                assert self.hp.post_pad.mel.has('length')
                if post_pad_dict is None:
                    post_pad_dict = dict()
                post_pad_dict['mel'] = int(self.hp.post_pad.mel.length *
                                           self.melspec.sample_rate)
                if self.preprocess_f0 is not None:
                    post_pad_dict['f0'] = post_pad_dict['mel']
            if self.hp.post_pad.has('cropped_mel'):
                assert self.hp.post_pad.cropped_mel.has('length')
                assert self.hp.has("random_crop_mel")
                if post_pad_dict is None:
                    post_pad_dict = dict()
                post_pad_dict['cropped_mel'] = int(
                    self.hp.post_pad.cropped_mel.length *
                    self.melspec.sample_rate)
            if self.hp.post_pad.has('cropped_mel_utt'):
                assert self.hp.post_pad.cropped_mel_utt.has('length')
                assert self.hp.has("random_crop_mel_utt")
                if post_pad_dict is None:
                    post_pad_dict = dict()
                post_pad_dict['cropped_mel_utt'] = int(
                    self.hp.post_pad.cropped_mel_utt.length *
                    self.melspec.sample_rate)
        return post_pad_dict


class DiscreteTokenDataset(MelSpecDataset):
    def __init__(self,
                 hp: Hparams,
                 hp_mel: Hparams,
                 hp_hubert: Hparams,
                 hp_rescale: Optional[Hparams] = None,
                 name: Optional[str] = None) -> None:
        assert hp.get("with_tokens", False)
        assert not hp.has("segment_size")
        assert not hp.has("truncate")
        super().__init__(hp, hp_mel, hp_rescale, name)
        self.deduplicate = hp_hubert.deduplicate
        self.token_sample_rate = hp_hubert.sample_rate

    def __getitem__(self, i):
        ret = super().__getitem__(i)
        tokens = self.tokens[i].long()
        if tokens.size(0) < ret['mel'].size(0):
            ret['mel'] = ret['mel'][: tokens.size(0)]
        if self.hp.has('token_segment_size'):
            min_crop_length = self.hp.token_segment_size
            if min_crop_length <= len(tokens):
                start_point = torch.randint(low=0,
                                            high=len(tokens)-min_crop_length+1,
                                            size=())
                tokens = tokens[start_point: start_point+min_crop_length]
                if self.preprocess_mels is None:
                    audio_start_point = int(float(start_point)
                                            / self.token_sample_rate
                                            * self.hp.sample_rate)
                    audio_crop_length = int(float(min_crop_length)
                                            / self.token_sample_rate
                                            * self.hp.sample_rate)
                    s = audio_start_point
                    e = audio_start_point + audio_crop_length
                    ret['audio'] = ret['audio'][s: e]
                mel_start_point = int(float(start_point)
                                      / self.token_sample_rate
                                      * self.melspec.sample_rate)
                mel_crop_length = int(float(min_crop_length)
                                      / self.token_sample_rate
                                      * self.melspec.sample_rate)
                s, e = mel_start_point, mel_start_point+mel_crop_length
                mel = pad_1d(ret['mel'], 1, e)
                ret['mel'] = mel[s: e]
                if self.preprocess_f0 is not None:
                    f0 = pad_1d(ret['f0'], 1, e)
                    ret['f0'] = f0[s: e]
        ret['tokens'] = tokens
        if self.deduplicate:
            output, inverse_indices, counts = torch.unique_consecutive(
                tokens,
                return_counts=True,
                return_inverse=True
            )
            ret['dedup_tokens'] = output
            ret['inverse_indices'] = inverse_indices
            ret['counts'] = counts
        return ret

    def get_post_pad_dict(self) -> Mapping[str, Any]:
        post_pad_dict = super().get_post_pad_dict()
        if self.hp.has('post_pad'):
            if self.hp.post_pad.has('tokens'):
                assert self.hp.post_pad.tokens.has('num_tokens')
                if post_pad_dict is None:
                    post_pad_dict = dict()
                if self.deduplicate:
                    post_pad_dict['dedup_tokens'] = (
                        self.hp.post_pad.tokens.num_tokens)
                else:
                    post_pad_dict['tokens'] = (
                        self.hp.post_pad.tokens.num_tokens)
        return post_pad_dict
