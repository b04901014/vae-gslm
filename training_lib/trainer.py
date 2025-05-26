from hparams.hp import Hparams
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils import data
from data.sampler import (StandardSampler, RandomBucketSampler,
                          ConcatLengthSampler)
from utils.helpers import move_data_to_device


class BaseTrainer(pl.LightningModule):
    def __init__(self, hp: Hparams) -> None:
        super().__init__()
        self.hp = hp
        hp.check_arg_in_hparams("model",
                                "data")
        self.gradient_update_step = 1
        if hp.has('training'):
            if hp.training.has("gradient_accumulation"):
                self.gradient_update_step = hp.training.gradient_accumulation
        self.compile_mode = None
        if hp.has("trainer"):
            if hp.trainer.has("compile"):
                if (hasattr(self.__class__, "_training_loop") and
                        callable(self._training_loop)):
                    self._training_loop = torch.compile(
                        self._training_loop,
                        mode=hp.trainer.compile.get("mode", "default"),
                        dynamic=hp.trainer.compile.get("dynamic", None)
                    )

    def load(self, strict: bool = True) -> None:
        self.hp.check_arg_in_hparams('pretrained_path')
        state_dict = torch.load(self.hp.pretrained_path)['state_dict']
        self.load_state_dict(state_dict, strict=strict)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return move_data_to_device(batch, device)

    def get_dataloader(self,
                       hp: Hparams,
                       dataset: data.Dataset) -> data.DataLoader:
        hp.check_arg_in_hparams("num_workers", "sampler")
        assert callable(getattr(dataset, "seqCollate", None))
        if self._trainer is None:
            world_size, rank = None, None
        else:
            world_size = self.trainer.world_size
            rank = self.trainer.local_rank
        if hp.sampler.type == "standard":
            hp.check_arg_in_hparams("batch_size")
            sampler = StandardSampler(dataset,
                                      shuffle=hp.sampler.shuffle,
                                      distributed=self.hp.trainer.distributed,
                                      drop_last=hp.sampler.get("drop_last",
                                                               True),
                                      world_size=world_size,
                                      rank=rank)
            dataloader = data.DataLoader(dataset,
                                         num_workers=hp.num_workers,
                                         batch_size=hp.batch_size,
                                         sampler=sampler,
                                         collate_fn=dataset.seqCollate,
                                         drop_last=hp.get("drop_last", True),
                                         pin_memory=True)
        elif hp.sampler.type == "bucket":
            hp.sampler.check_arg_in_hparams("num_buckets")
            batch_size, batch_length = None, None
            if hp.get("batch_size", False):
                batch_size = hp.batch_size
            elif hp.get("batch_length", False):
                batch_length = hp.batch_length
            else:
                raise ValueError("Must present one of"
                                 " batch_size or batch_length")
            sampler = RandomBucketSampler(hp.sampler.num_buckets,
                                          dataset.lengths,
                                          batch_size,
                                          batch_length,
                                          hp.sampler.get("drop_last", False),
                                          self.hp.trainer.distributed,
                                          world_size=world_size,
                                          rank=rank)
            dataloader = data.DataLoader(dataset,
                                         num_workers=hp.num_workers,
                                         batch_sampler=sampler,
                                         collate_fn=dataset.seqCollate,
                                         drop_last=hp.get("drop_last", True),
                                         pin_memory=True)
        elif hp.sampler.type == "concat":
            hp.check_arg_in_hparams("batch_size", "length")
            if hp.get("with_text", False):
                hp.check_arg_in_hparams("text_max_length")
            sampler = ConcatLengthSampler(
                batch_size=batch_size,
                max_length=hp.length,
                length=dataset.lengths,
                text_max_length=hp.get("text_max_length", None),
                distributed=self.hp.trainer.distributed,
                world_size=world_size,
                rank=rank)
            # TODO: Finish Concatenation Sampler
            dataloader = data.DataLoader(dataset,
                                         num_workers=hp.num_workers,
                                         batch_sampler=sampler,
                                         collate_fn=sampler.seqCollate,
                                         pin_memory=True)
        else:
            raise NotImplementedError("Currently only `standard` and `bucket`"
                                      "is supported as sampler.")
        return dataloader

    def init_weights(self, module) -> None:
        init_std = self.hp.training.get("init_std", 1.0)
        if getattr(module, 'bias', None) is not None:
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        init_op = getattr(module, "custom_weight_init", None)
        if callable(init_op):
            init_op(init_std)
