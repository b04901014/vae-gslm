# VAE-GSLM
- Official implementation for the paper: A Variational Framework for Improving Naturalness in Generative Spoken Language Models
- Audio samples used for MOS evaluation can be accessed [here](https://cmu.box.com/s/irzoqoj4za0gww3sl7nby18flxcb9ruj) 

## Setup the environment

1. Setup conda environment:
```
conda create --name vae-gslm python=3.9
conda activate vae-gslm
```

2. Install [faiss](https://github.com/facebookresearch/faiss) for getting semantic tokens:
```
conda install -c conda-forge faiss-gpu=1.9.0
```

3. Install the required packages
```
pip3 install -r requirements.txt
```

If you wish to only use the pre-trained model, jump to the section [Pre-trained Models](#pre-trained-models) section. 

If you wish to train everything from scratch, jump to the [Training from scratch](#training-from-scratch) section.

We also provide intermediate checkpoints: 

## Training from scratch
Here we use LibriSpeech-960 and Libri-light 60k for example, feel free to change to your own dataset by changing the paths.

- In this example, we use LibriSpeech-960 to train the Hifi-GAN vocoder and as prompt for the SpeechLM; Libri-light 60k is used to train the SpeechLM. 
- Get the [LibriSpeech-960](https://www.openslr.org/12) downloaded into `./LibriSpeech-960`.
- Make sure you have [Libri-light](https://github.com/facebookresearch/libri-light/tree/main/data_preparation) downloaded into `./ll60`.
- Segment Libri-light using their official code (we use 20s segments), and make the wavfiles locate at `./ll60/vad_20s`

1. Pre-process the datasets:
 - Produce the list of files in LibriSpeech
```
cd ./LibriSpeech-960/dev/
find . -name "*.flac" > metadata.txt
cd ./LibriSpeech-960/train/
find . -name "*.flac" > metadata.txt
```
 - Produce the list of files in Libri-light
```
cd ./ll60/vad_20s
find . -name "*.flac" > metadata.txt
```

2. Train the vocoder: 
 - `mkdir ./vocoder_ckpt`
 - Run `python -m scripts.train -c configs/train/vocoder/hfgan_16k_50hz_librispeech.yaml`
 - Get the config file (e.g., `outputs/hfgan_50hz_librispeech/log/version_*/hp.yaml`) and move it to `./vocoder_ckpt`
 - Get final checkpoint (e.g., `./outputs/hfgan_50hz_librispeech/ckpt/version_*/epoch\=***-cpt.ckpt`), rename the checkpoint as `last.ckpt` and move it to `./vocoder_ckpt`

2. Get the semantic tokens:
 - `python -m scripts.hubert_kmeans -i ./ll60/vad_20s/metadata.txt -w ./ll60/vad_20s/ -o ./ll60/vad_20s/token.txt`
 - `python -m scripts.hubert_kmeans -c ./ll60/vad_20s/kmeans_v200.npy -i ./LibriSpeech-960/dev/metadata.txt -w ./LibriSpeech-960/dev/ -o ./LibriSpeech-960/dev/token.txt`

 `-c` sepcifies the centroids that are trained in the first step on Libri-light.

3. Preprocess Mel-spectrograms for faster training:
 - `python -m scripts.preprocess_mels -c configs/preprocess/hfgan_16k_50hz_libri-light.yaml -o ./ll60/vad_20s/mels`

4. Train the VAE-GSLM:
 - `python -m scripts.train -c configs/train/speech/vae-gslm.yaml`
 - Get the config file (e.g., `./outputs/vae-gslm/log/version_*/hp.yaml`) and move it to `./vae-gslm_ckpt`
 - Get final checkpoint (e.g., `./outputs/vae-gslm/ckpt/version_*/epoch\=***-cpt.ckpt`), rename the checkpoint as `last.ckpt` and move it to `./vae-gslm_ckpt`

## Pre-trained Models
- [HiFi-GAN Vocoder trained on LibriSpeech](https://cmu.box.com/s/tp53v4wnfue4qkyhif7rk7uc4u9m6lqr)
- [VAE-GSLM trained on Libri-light](https://cmu.box.com/s/vtfu43d6rkcishfb1q8gixuy3zkc39wj)
- [K-means Clustering Centroids for Semantic Tokens](https://cmu.box.com/s/n2aehj837a6ivdz5mmrx0tpc49zhqxt3)

Put the vocoder checkpoint files in `./vocoder_ckpt`, SpeechLM checkpoint files in `./vae-gslm_ckpt`.
Put the clustering centroids as `./ll60/vad_20s/kmeans_v200.npy`.

## Running Inference
If you haven't done so, run:
 - `python -m scripts.hubert_kmeans -c ./ll60/vad_20s/kmeans_v200.npy -i ./LibriSpeech-960/dev/metadata.txt -w ./LibriSpeech-960/dev/ -o ./LibriSpeech-960/dev/token.txt`

to get the semantic tokens for the prompts.

Then run inference:
- `python -m scripts.infer -c configs/vae-gslm.yaml`

This script uses the tokens specified in `./LibriSpeech-960/dev/token.txt`, get the first 3 second as prompt and the model generates 10 seconds of continuation. Then we use VAD to trim the excessive silence. The samples will be in `./samples`. Feel free the adjust the inference parameters in `configs/vae-gslm.yaml`.
