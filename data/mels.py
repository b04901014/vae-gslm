import torch
from typing import Optional, Callable, Union
from torchaudio import functional as F
from torch import Tensor


def _get_spec_norms(normalized: Union[str, bool]):
    frame_length_norm, window_norm = False, False
    if torch.jit.isinstance(normalized, str):
        if normalized not in ["frame_length", "window"]:
            raise ValueError("Invalid normalized parameter: {}".format(normalized))
        if normalized == "frame_length":
            frame_length_norm = True
        elif normalized == "window":
            window_norm = True
    elif torch.jit.isinstance(normalized, bool):
        if normalized:
            window_norm = True
    else:
        raise TypeError("Input type not supported")
    return frame_length_norm, window_norm


def spectrogram(
    waveform: Tensor,
    pad: int,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    normalized: Union[bool, str],
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    return_complex: Optional[bool] = None,
) -> Tensor:
    if pad > 0:
        # TODO add "with torch.no_grad():" back when JIT supports it
        waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

    frame_length_norm, window_norm = False, False#_get_spec_norms(normalized)

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    # default values are consistent with librosa.core.spectrum._spectrogram
    spec_f = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=frame_length_norm,
        onesided=onesided,
        return_complex=return_complex,
    )
    if window_norm:
        spec_f /= window.pow(2.0).sum().sqrt()
    return spec_f


class MagnitudeSpectrogram(torch.nn.Module):
    __constants__ = ["n_fft", "win_length", "hop_length",
                     "pad", "power", "normalized"]

    def __init__(
        self,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad: int = 0,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: Optional[float] = 2.0,
        normalized: Union[bool, str] = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        return_complex: Optional[bool] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once("torchaudio.transforms.Spectrogram")
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequencies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window)
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.return_complex = return_complex

    def forward(self, waveform: Tensor) -> Tensor:
        spec = spectrogram(
            waveform,
            self.pad,
            self.window,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.normalized,
            self.center,
            self.pad_mode,
            self.onesided,
            self.return_complex
        )
        if self.return_complex:
            return torch.abs(spec)
        return torch.sqrt(spec.pow(self.power).sum(-1)+(1e-9))

class MelScale(torch.nn.Module):
    __constants__ = ["n_mels", "sample_rate", "f_min", "f_max"]

    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: int = 201,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.norm = norm
        self.mel_scale = mel_scale

        if f_min > self.f_max:
            raise ValueError("Require f_min: {} <= f_max: {}".format(f_min, self.f_max))
        fb = F.melscale_fbanks(n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.norm, self.mel_scale)
        self.register_buffer("fb", fb)

    def forward(self, specgram: Tensor) -> Tensor:
        mel_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)
        return mel_specgram

class MelSpectrogram(torch.nn.Module):
    __constants__ = ["sample_rate", "n_fft", "win_length", "hop_length", "pad", "n_mels", "f_min"]

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: Optional[bool] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        use_complex: bool = True
    ) -> None:
        super(MelSpectrogram, self).__init__()
        torch._C._log_api_usage_once("torchaudio.transforms.MelSpectrogram")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min
        self.spectrogram = MagnitudeSpectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=window_fn,
            power=self.power,
            normalized=self.normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=True,
            return_complex=use_complex
        )
        self.mel_scale = MelScale(
            self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1, norm, mel_scale
        )

    def forward(self, waveform: Tensor) -> Tensor:
        specgram = self.spectrogram(waveform)
        mel_specgram = self.mel_scale(specgram)
        return mel_specgram

