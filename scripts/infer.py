from hparams.hp import Hparams
import argparse
import importlib
import torch
from pathlib import Path
import logging
from lightning.pytorch import Trainer
import os
from utils.helpers import get_last_ckpt

parser = argparse.ArgumentParser(prog="Infer a model with a given config")
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-v', '--version', type=str, default=None)
parser.add_argument('-log', '--loglevel', type=str, default='WARNING',
                    choices=logging._nameToLevel.keys())
args = parser.parse_args()

logging.basicConfig(level=args.loglevel.upper())

hp = Hparams.from_yamlfile(args.config)
if hp.has("output_dir"):
    Path(hp.output_dir).mkdir(parents=True, exist_ok=True)

if args.version is not None:
    hp.check_arg_in_hparams("exp_dir")
    tmp_ckpt_dir = "./tmp_ckpt_infer"
    if args.version is not None:
        tmp_ckpt_dir += f"_{args.version}"
    logging.info(f"Creating temporary directory {tmp_ckpt_dir}...")
    Path(tmp_ckpt_dir).mkdir(parents=True, exist_ok=True)
    exp_path = os.path.join(hp.exp_dir, "ckpt", f"version_{args.version}")
    last_ckpt = get_last_ckpt(exp_path)
    print(last_ckpt)
    hp_path = os.path.join(hp.exp_dir, "log",
                           f"version_{args.version}", "hp.yaml")
    Path(os.path.join(tmp_ckpt_dir, "last-cpt.ckpt")).symlink_to(
        Path(last_ckpt).resolve())
    if 'symbols.json' in os.listdir(exp_path):
        Path(os.path.join(tmp_ckpt_dir, "symbols.json")).symlink_to(
            Path(os.path.join(exp_path, 'symbols.json')).resolve())
    Path(os.path.join(tmp_ckpt_dir, "hp.yaml")).symlink_to(
        Path(hp_path).resolve())
    logging.info(f"Will load from {last_ckpt} and {hp_path}...")
    hp.ckpt_path = tmp_ckpt_dir

torch.set_float32_matmul_precision('high')

p, m = hp.identifier.rsplit('.', 1)
model = getattr(importlib.import_module(p), m, None)
if model is None:
    raise ValueError(f"{m} not found in {p}.")
model = model(hp)

if args.version is not None:
    logging.info("Cleaning up temporary directory...")
    Path(os.path.join(tmp_ckpt_dir, "last-cpt.ckpt")).unlink()
    Path(os.path.join(tmp_ckpt_dir, "hp.yaml")).unlink()
    if 'symbols.json' in os.listdir(exp_path):
        Path(os.path.join(tmp_ckpt_dir, "symbols.json")).unlink()
    Path(tmp_ckpt_dir).rmdir()

trainer = Trainer(
    precision=hp.precision,
    devices="auto"
)
trainer.test(model)
