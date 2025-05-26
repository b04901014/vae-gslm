from __future__ import annotations
from types import SimpleNamespace
from argparse import Namespace
from typing import Mapping, Any
import json
import yaml


class Hparams(SimpleNamespace):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def check_arg_in_hparams(self, *args):
        for arg in args:
            if arg not in self.__dict__:
                raise ValueError(f"{arg} not specifed"
                                 f" in the hyperapramer: {self}")

    def merge(self, hp: Hparams) -> Hparams:
        return Hparams(**self.__dict__, **hp.__dict__)

    def get(self, x: str, default=None):
        if x in self.__dict__:
            return self.__dict__[x]
        return default

    def has(self, x: str):
        return x in self.__dict__

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return repr(self.__dict__)

    def to_dict(self) -> Mapping[str, Any]:
        return json.loads(json.dumps(self,
                                     default=lambda o: o.__dict__))

    @classmethod
    def from_jsonfile(cls, jsonfile: str) -> Hparams:
        with open(jsonfile, 'r') as f:
            return json.load(f,
                             object_hook=lambda x: Hparams(**x))

    @classmethod
    def from_json(cls, json_s: str) -> Hparams:
        return json.loads(json_s,
                          object_hook=lambda x: Hparams(**x))

    @classmethod
    def from_argparse(cls, args: Namespace) -> Hparams:
        return json.loads(json.dumps(args.__dict__),
                          object_hook=lambda x: Hparams(**x))

    @classmethod
    def from_yamlfile(cls, yamlfile: str) -> Hparams:
        with open(yamlfile, 'r') as f:
            data = yaml.safe_load(f)
            return json.loads(json.dumps(data),
                              object_hook=lambda x: Hparams(**x))

    def save(self, path: str) -> None:
        """Save the Hparams to yaml file """
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f)
