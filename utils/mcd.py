import pysptk
import numpy as np


def wav2mcep(x, mcep_dim=23, mcep_alpha=0.42, n_fft=1024, n_shift=256):
    win = pysptk.sptk.hamming(n_fft)
    n_frame = (len(x) - n_fft) // n_shift + 1

    def window(i):
        return pysptk.sptk.mcep(
                        x[n_shift * i: n_shift * i + n_fft] * win,
                        mcep_dim,
                        mcep_alpha,
                        eps=1e-8,
                        etype=1,
        )
    mcep = list(map(window, range(n_frame)))
    return np.stack(mcep)


def mcd(a, b):
    a, b = wav2mcep(a), wav2mcep(b)
    diff2sum = np.sum((a - b) ** 2, 1)
    return np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum))
