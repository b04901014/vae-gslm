from hparams.hp import Hparams
from typing import Optional
from .absolute import SinCos
from .rotary import Rotary
from .alibi import ALiBi
from .t5 import T5RPE


def get_positional_encoding(name: str,
                            hp: Hparams,
                            ndim: Optional[int] = None,
                            nheads: Optional[int] = None):
    if name == "SinCos":
        assert ndim is not None
        return SinCos(ndim,
                      hp.get("maxpos", 10000),
                      hp.get("fixed_pos", False),
                      hp.get("scaled", False))
    elif name == "Rotery":
        assert ndim is not None
        return Rotary(ndim,
                      theta=hp.get("theta", 10000),
                      max_freq=hp.get("max_freq", 10),
                      num_freqs=hp.get("num_freqs", 1),
                      learned_freq=hp.get("learned_freq", False),
                      use_xpos=hp.get("use_xpos", False))
    elif name == "ALiBi":
        assert nheads is not None
        return ALiBi(nheads, hp.get("maxpos", 10000))
    elif name == "T5RPE":
        assert nheads is not None
        hp.check_arg_in_hparams("bidirectional",
                                "num_buckets",
                                "max_distance")
        return T5RPE(nheads,
                     hp.bidirectional,
                     hp.num_buckets,
                     hp.max_distance)
    else:
        raise ValueError(f"{name} is not a valid PE type.")
