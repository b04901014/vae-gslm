import argparse
import torchaudio
import os
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import faiss
import numpy as np
import random
from pathlib import Path
import torch

parser = argparse.ArgumentParser(
    prog="Run HuBERT tokenization on a parsed metadata")
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('-w', '--wavdir', type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-m', '--model', type=str,
                    default="facebook/hubert-large-ll60k")
parser.add_argument('-c', '--centroids', type=str, default=None)
parser.add_argument('-d', '--delimiter', type=str, default=" ")
parser.add_argument('-v', '--vocab', type=int, default=200)
parser.add_argument('-p', '--percentage', type=float, default=0.1)
parser.add_argument('-s', '--size_max', type=float, default=60)
args = parser.parse_args()


def load_dataset(metadata: str):
    lines = []
    fns = []
    with open(metadata, 'r', errors='ignore') as f:
        for line in f.readlines():
            if not line.strip():
                continue
            fn = line.strip().split('|')
            lines.append(line.strip())
            fns.append(fn[0])
    return lines, fns


model = HubertModel.from_pretrained(args.model,
                                    torch_dtype=torch.float16)
model.cuda()
model.eval()
processor = Wav2Vec2FeatureExtractor()

dataset = list(zip(*load_dataset(args.input)))
subset = random.sample(dataset,
                       int(len(dataset)*args.percentage))

if args.centroids is None:
    kmvs = []
    # Run k-means
    for line, fn in tqdm(subset):
        fn = os.path.join(args.wavdir, fn)
        audio, sr = torchaudio.load(fn)
        assert sr == 16000
        if audio.size(-1) / float(sr) >= args.size_max:
            continue
        audio = processor(audio[0], return_tensors="pt",
                          sampling_rate=sr).input_values.half()
        out = model(audio.cuda()).last_hidden_state
        kmvs.append(out.detach().cpu().numpy()[0])
    kmvs = np.concatenate(kmvs, 0)

    print("Training Kmeans...")
    niter = 20
    verbose = True
    kmeans = faiss.Kmeans(kmvs.shape[-1], args.vocab,
                          niter=niter, verbose=verbose)
    kmeans.train(kmvs)
    _, indicies = kmeans.index.search(kmvs, 1)
    index_fn = Path(args.output).parents[0] / f"kmeans_v{args.vocab}.npy"
    with open(index_fn, 'wb') as f:
        np.save(f, kmeans.centroids)
else:
    # Build kmeans to run inference with given centroids
    centroids = np.load(args.centroids)
    kmeans = faiss.Kmeans(centroids.shape[-1], args.vocab)
    kmeans.centroids = centroids
    kmeans.index.reset()
    kmeans.index.add(kmeans.centroids)

with open(args.output, 'w') as f:
    i = 0
    for line, fn in tqdm(dataset):
        fn = os.path.join(args.wavdir, fn)
        audio, sr = torchaudio.load(fn)
        assert sr == 16000
        if audio.size(-1) / float(sr) >= args.size_max:
            continue
        audio = processor(audio[0], return_tensors="pt",
                          sampling_rate=sr).input_values.half()
        out = model(audio.cuda()).last_hidden_state
        _, indicies = kmeans.index.search(
            out.detach().cpu().numpy()[0], 1)
        out = [str(o[0]) for o in indicies]
        out = args.delimiter.join(out)
        f.write(f'{line}|{out}\n')
        i += 1
