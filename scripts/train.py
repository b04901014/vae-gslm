from hparams.hp import Hparams
import argparse
import importlib
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from training_lib.callbacks import CompactModelCheckpoint
import torch
import os
from pathlib import Path
import logging

parser = argparse.ArgumentParser(prog="Training a model with a given config")
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-n', '--name', type=str, default=None)
parser.add_argument('-p', '--profile', action="store_true")
parser.add_argument('-s', '--sanity', action="store_true")
parser.add_argument('-d', '--detect_anomaly', action="store_true")
parser.add_argument('-r', '--resume_checkpoint', type=str, default=None)
parser.add_argument('-v', '--version', type=int, default=None)
parser.add_argument('-log', '--loglevel', type=str, default='WARNING',
                    choices=logging._nameToLevel.keys())
args = parser.parse_args()

logging.basicConfig(level=args.loglevel.upper())

hp = Hparams.from_yamlfile(args.config)
hp.check_arg_in_hparams("trainer", "logging")
hp.logging.check_arg_in_hparams("log_dir")
hp.trainer.check_arg_in_hparams("save_every_n_epoch",
                                "precision",
                                "distributed",
                                "identifier",
                                "total_steps")
if not hp.trainer.has("check_val_every_n_epoch"):
    hp.trainer.check_arg_in_hparams("val_check_interval")

if hp.has("ulimit"):
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hp.ulimit, rlimit[1]))

torch.set_float32_matmul_precision('high')
if hp.trainer.distributed:
    hp.trainer.check_arg_in_hparams("ddp_strategy")

Path(hp.logging.log_dir).mkdir(parents=True, exist_ok=True)

p, m = hp.trainer.identifier.rsplit('.', 1)
model = getattr(importlib.import_module(p), m, None)
if model is None:
    raise ValueError(f"{m} not found in {p}.")
model = model(hp)

logging_dir = os.path.join(hp.logging.log_dir, 'log')
logger = TensorBoardLogger(logging_dir, name=args.name, version=args.version)
ckpt_dir = logger.version
if isinstance(ckpt_dir, int):
    ckpt_dir = f"version_{ckpt_dir}"
ckpt_dir = os.path.join(hp.logging.log_dir, 'ckpt', ckpt_dir)
checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_dir,
    filename='{epoch}-{step}',
    every_n_epochs=hp.trainer.save_every_n_epoch,
    monitor='step',
    mode="max",
    save_top_k=hp.trainer.get("save_top_k", 5)
)
compact_checkpoint_callback = CompactModelCheckpoint(
    dirpath=ckpt_dir,
    filename='{epoch}-{step}-cpt',
    every_n_epochs=hp.trainer.save_every_n_epoch,
    monitor='step',
    mode="max",
    save_top_k=hp.trainer.get("save_top_k", 5)
)

acc_grad = hp.trainer.get("accumulate_grad_batches", 1)
val_check_interval = hp.trainer.get("val_check_interval", 1.0)
if isinstance(val_check_interval, int):
    if hp.has("training"):
        val_check_interval *= hp.training.get("gradient_accumulation", 1)
        val_check_interval = int(val_check_interval)

trainer = Trainer(
    precision=hp.trainer.precision,
    callbacks=[checkpoint_callback,
               compact_checkpoint_callback,
               TQDMProgressBar(refresh_rate=50)],
    num_sanity_val_steps=8 if args.sanity else 0,
    max_steps=2000 if args.profile else hp.trainer.total_steps,
    devices=("auto" if hp.trainer.distributed else 1),
    strategy=(hp.trainer.ddp_strategy if hp.trainer.distributed else 'auto'),
    use_distributed_sampler=False,
    accumulate_grad_batches=acc_grad,
    logger=logger,
    check_val_every_n_epoch=hp.trainer.get("check_val_every_n_epoch", None),
    val_check_interval=val_check_interval,
    limit_val_batches=hp.trainer.get("limit_val_batches", 1.0),
    profiler=("simple" if args.profile else None),
    detect_anomaly=args.detect_anomaly
)
trainer.fit(model, ckpt_path=args.resume_checkpoint)
