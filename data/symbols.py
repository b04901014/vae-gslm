from __future__ import annotations
from typing import List, Union, Set
from collections.abc import Iterable
import json


class Symbols(object):
    def __init__(self, x: Set, delim: str):
        """
        `x` is a list of symbols to initiate.
        It appends <unk> to encode any symbols that
        are out of `x`
        It adds <pad> for paddings, <sos> for start.
        """
        self._symbol = x
        self.symbol = list(sorted(self._symbol))
        self.symbol.append('<unk>')
        self.symbol = ['<pad>', '<sos>'] + self.symbol
        self.mapping = {
            k: v
            for v, k in enumerate(self.symbol)
        }
        self.delimiter = delim

    @property
    def pad_idx(self):
        return self.mapping['<pad>']

    @property
    def sos_idx(self):
        return self.mapping['<sos>']

    @property
    def unk_idx(self):
        return self.mapping['<unk>']

    @property
    def num_symbols(self):
        return len(self.symbol)

    def encode(self,
               x: Union[str, List[str]]
               ) -> List[int]:
        if isinstance(x, str):
            x = x.split(self.delimiter)
        x = [self.mapping.get(s, self.unk_idx) for s in x]
        return [self.sos_idx] + x

    def decode(self,
               x: Iterable[int]) -> str:
        return self.delimiter.join([self.symbol[e]
                                    for e in x if e != self.sos_idx])

    def save(self, path: str) -> None:
        d = {
            'symbols': list(self._symbol),
            'mapping': self.mapping,
            'delimiter': self.delimiter
        }
        with open(path, 'w') as f:
            json.dump(d, f)

    @classmethod
    def load(cls, path: str) -> Symbols:
        with open(path, 'r') as f:
            _symbol = json.load(f)
            _symbol, _delim = _symbol['symbols'], _symbol['delimiter']
        return Symbols(_symbol, _delim)
