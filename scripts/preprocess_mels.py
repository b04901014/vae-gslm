from data.dataset import MelSpecDataset
from hparams.hp import Hparams
from torch.utils import data
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(prog="Preprocess mels for a given directory")
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)
args = parser.parse_args()

hp = Hparams.from_yamlfile(args.config)

dataset = MelSpecDataset(hp.data, hp.mel)

dataloader = data.DataLoader(dataset,
                             num_workers=hp.data.num_workers,
                             batch_size=1,
                             shuffle=False)

i = 0
ll = len(hp.data.wavdir)
for d in tqdm(dataloader):
    p = Path(dataset.audios[i][ll:])
    p = Path(args.output) / p
    p.parents[0].mkdir(parents=True, exist_ok=True)
    fname = str(p.parents[0] / Path(p.stem + '.npy'))
    np.save(fname, d['mel'][0])
    i += 1
