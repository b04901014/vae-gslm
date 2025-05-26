import torch
from hparams.hp import Hparams
import os
import importlib
from training_lib.trainer import BaseTrainer


class BaseInferer(BaseTrainer):
    def __init__(self, hp: Hparams, *args, **kwargs):
        super().__init__(hp)
        hp.check_arg_in_hparams("ckpt_path")
        hp_model = Hparams.from_yamlfile(
            os.path.join(hp.ckpt_path, 'hp.yaml')
        )
        self.hp_model = hp_model

    def load_model(self, *args, **kwargs):
        p, m = self.hp.model.identifier.rsplit('.', 1)
        model = getattr(importlib.import_module(p), m, None)
        if model is None:
            raise ValueError(f"{m} not found in {p}.")
        model = model(self.hp_model.model, *args, **kwargs)
        model.load_state_dict(
            torch.load(os.path.join(self.hp.ckpt_path, 'last-cpt.ckpt'),
                       map_location='cpu'),
            strict=False
        )
        self.model = model

    def test_dataloader(self):
        pass

    def test_step(self):
        pass
