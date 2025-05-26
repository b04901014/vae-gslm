# Wrapper of "https://github.com/maxrmorrison/torchcrepe/blob/master/torchcrepe/core.py"
import torchcrepe
import warnings
import torch
from torchcrepe.core import (PITCH_BINS,
                             SAMPLE_RATE, WINDOW_SIZE,
                             postprocess)

def preprocess(audio,
               sample_rate,
               hop_length=None,
               batch_size=None,
               device='cpu',
               pad=True):
    # Default hop length of 10 ms
    hop_length = sample_rate // 100 if hop_length is None else hop_length

    # Resample
    if sample_rate != SAMPLE_RATE:
        audio = resample(audio, sample_rate)
        hop_length = int(hop_length * SAMPLE_RATE / sample_rate)

    # Get total number of frames

    # Maybe pad
    if pad:
        total_frames = 1 + int(audio.size(1) // hop_length)
        audio = torch.nn.functional.pad(
            audio,
            (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    else:
        total_frames = 1 + int((audio.size(1) - WINDOW_SIZE) // hop_length)

    # Default to running all frames in a single batch
    batch_size = total_frames if batch_size is None else batch_size

    # Generate batches
    for i in range(0, total_frames, batch_size):

        # Batch indices
        start = max(0, i * hop_length)
        end = min(audio.size(1),
                  (i + batch_size - 1) * hop_length + WINDOW_SIZE)
        # Chunk
        frames = torch.nn.functional.unfold(
            audio[:, None, None, start:end],
            kernel_size=(1, WINDOW_SIZE),
            stride=(1, hop_length))

        # shape=(1 + int(time / hop_length, 1024)
        frames = frames.transpose(1, 2).reshape(-1, WINDOW_SIZE)

        # Mean-center
        frames -= frames.mean(dim=1, keepdim=True)

        # Scale
        # Note: during silent frames, this produces very large values. But
        # this seems to be what the network expects.
        frames /= torch.max(torch.tensor(1e-10, device=frames.device),
                            frames.std(dim=1, keepdim=True))

        yield frames

def predict(audio,
            sample_rate,
            model,
            device,
            hop_length=None,
            fmin=50.,
            fmax=550.,
            decoder=torchcrepe.decode.viterbi,
            return_harmonicity=False,
            return_periodicity=False,
            batch_size=None,
            pad=True):
    # Deprecate return_harmonicity
    if return_harmonicity:
        message = (
            'The torchcrepe return_harmonicity argument is deprecated and '
            'will be removed in a future release. Please use '
            'return_periodicity. Rationale: if network confidence measured '
            'harmonics, the value would be low for non-harmonic, periodic '
            'sounds (e.g., sine waves). But this is not observed.')
        warnings.warn(message, DeprecationWarning)
        return_periodicity = return_harmonicity

    results = []

    # Postprocessing breaks gradients, so just don't compute them
    with torch.no_grad():

        # Preprocess audio
        generator = preprocess(audio,
                               sample_rate,
                               hop_length,
                               batch_size,
                               device,
                               pad)
        for frames in generator:

            # Infer independent probabilities for each pitch bin
            probabilities = model(frames, embed=False)

            # shape=(batch, 360, time / hop_length)
            probabilities = probabilities.reshape(
                audio.size(0), -1, PITCH_BINS).transpose(1, 2)

            # Convert probabilities to F0 and periodicity
            result = postprocess(probabilities,
                                 fmin,
                                 fmax,
                                 decoder,
                                 return_harmonicity,
                                 return_periodicity)

            # Place on same device as audio to allow very long inputs
            if isinstance(result, tuple):
                result = (result[0].to(audio.device),
                          result[1].to(audio.device))
            else:
                result = result.to(audio.device)

            results.append(result)

    # Split pitch and periodicity
    if return_periodicity:
        pitch, periodicity = zip(*results)
        return torch.cat(pitch, 1), torch.cat(periodicity, 1)

    # Concatenate
    return torch.cat(results, 1)
