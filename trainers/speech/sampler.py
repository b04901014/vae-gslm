import torch.nn as nn
import torch
from typing import Mapping, Optional, Tuple
from utils.tensormask import TensorMask


class ARTRSampler(object):
    def __init__(self, model: nn.Module):
        self.model = model
        self.has_utterance = (
            hasattr(self.model, "utterance_encoder") and
            self.model.utterance_encoder is not None
        )
        self.model_use_tokens = (
            hasattr(self.model, 'use_tokens') and self.model.use_tokens)

    def __call__(self,
                 length: int,
                 prior: torch.Tensor,
                 temperature: float = 1.0,
                 token_temperature: float = 1.0,
                 truncated_norm: Optional[Tuple[float, float]] = None,
                 return_attn: bool = False,
                 encoder_temperature: float = 1.0
                 ) -> Mapping:
        if self.has_utterance:
            u_c = self.model.encode_utterance(TensorMask(prior))
        prior = self.model.encode(TensorMask(prior),
                                  temperature=encoder_temperature).value
        state = self.model.initial_state(
            prior.shape[0], device=prior.device
        )
        if hasattr(self.model, 'f0') and self.model.f0 is not None:
            f0_state = torch.zeros((prior.shape[0], 1, 1), dtype=state.dtype,
                                   device=state.device)
            state = torch.cat([state[..., None], f0_state], -1)
        if self.model_use_tokens:
            state = prior
        else:
            state = torch.cat([state, prior], 1)
        iters = {
            'output': state,
            'kv': None
        }
        outputs = {
            'output': [prior]
        }
        if return_attn:
            outputs['attn'] = []
        for i in range(length):
            iters = self.model.step(
                iters['output'],
                temperature=temperature,
                token_temperature=token_temperature,
                truncated_norm=truncated_norm,
                past_kv=iters['kv'],
                return_attn=return_attn,
                push_init_state=(i == 0 and self.model_use_tokens)
            )
            if i == 0:
                iters['output'] = iters['output'][:, -1:]
            outputs['output'].append(iters['output'])
        if self.has_utterance:
            outputs['output'] = self.model.decode(
                TensorMask(torch.cat(outputs['output'], 1)),
                u_c=u_c
            )
        else:
            outputs['output'] = self.model.decode(
                TensorMask(torch.cat(outputs['output'], 1))
            )
        return outputs
