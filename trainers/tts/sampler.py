import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Mapping
from utils.tensormask import TensorMask


class ARTRTTSSampler(object):
    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def __call__(self,
                 text: TensorMask,
                 spkr: torch.Tensor,
                 max_frames: int,
                 min_frames: int,
                 temperature: float = 1.0,
                 eos_threshold: float = 0.5,
                 return_attn: bool = False
                 ) -> Mapping:
        batch_size, _ = text.size()
        condition, spkr = self.model.encode_condition(text, spkr,
                                                      return_attn=return_attn)
        outputs = dict()
        if return_attn:
            condition, text_attn = condition
            text_attn = torch.stack(text_attn, 0).transpose(0, 1)
            outputs['text_self_attn'] = self.truncate_attention(
                text_attn, text.length, text.length)
            outputs['cross_attn'] = []
            outputs['self_attn'] = []
        original_condition = condition
        state = self.model.initial_state(
            batch_size, device=condition.device
        )
        iters = {
            'output': state,
            'kv': None
        }
        batch_indicator = torch.ones(batch_size,
                                     device=text.device, dtype=torch.bool)
        batch_idx = torch.arange(batch_size, device=text.device)

        for i in range(max_frames):
            iters = self.model.step(
                iters['output'],
                condition,
                spkr,
                temperature=temperature,
                past_kv=iters['kv'],
                eos_threshold=eos_threshold,
                return_attn=return_attn
            )
            if return_attn:
                outputs['self_attn'].append(
                    self.process_attention(iters['self_attn'], batch_indicator)
                )
                outputs['cross_attn'].append(
                    self.process_attention(iters['cross_attn'],
                                           batch_indicator)
                )
            if i == 0:
                outputs['output'] = [(iters['output'],
                                      batch_indicator.clone())]
            elif i < min_frames:
                outputs['output'].append((iters['output'],
                                          batch_indicator.clone()))
            else:
                step_value = torch.zeros([batch_size, 1,
                                          iters['output'].size(-1)],
                                         device=text.device)
                step_value[batch_indicator] = iters['output']
                outputs['output'].append(
                    (step_value.clone(), batch_indicator.clone())
                )
                eos = iters['eos']
                eos_batch_idx = batch_idx[batch_indicator][eos]
                batch_indicator[eos_batch_idx] = False
                iters['output'] = iters['output'][~eos]
                new_kv = []
                for kv_pair in iters['kv']:
                    k = kv_pair['key'][~eos]
                    v = kv_pair['value'][~eos]
                    new_kv.append({'key': k, 'value': v})
                iters['kv'] = new_kv
                condition = TensorMask(condition.value[~eos],
                                       condition.mask[~eos])
                spkr = spkr[~eos]
            if torch.all(~batch_indicator).item():
                break
        tensor, mask = zip(*outputs['output'])
        tensor, mask = torch.cat(tensor, 1), torch.stack(mask, 1)
        out = TensorMask(tensor, mask)
        outputs['output'] = self.model.decode(
            out, original_condition)
        if return_attn:
            outputs['self_attn'] = self.aggregate_attention(
                outputs['self_attn'], out.length, out.length)
            outputs['cross_attn'] = self.aggregate_attention(
                outputs['cross_attn'], out.length, original_condition.length)
        return outputs

    def aggregate_attention(self, x, q_length, kv_length):
        ret = []
        for layers in zip(*x):
            max_length = max([layer.size(-1) for layer in layers])
            layers = [F.pad(layer, (0, max_length - layer.size(-1)))
                      for layer in layers]
            ret.append(torch.cat(layers, 2))
        ret = torch.stack(ret, 0)
        ret = ret.transpose(0, 1)
        return self.truncate_attention(ret, q_length, kv_length)

    def truncate_attention(self, x, q_length, kv_length):
        output = []
        for i, batch in enumerate(x):
            output.append(batch[:, :, :q_length[i], :kv_length[i]])
        return output

    def process_attention(self, x, batch_indicator):
        ret = []
        for attn in x:
            shape = list(attn.size())
            shape[0] = batch_indicator.size(0)
            out = torch.zeros(shape, device=attn.device)
            out[batch_indicator] = attn
            ret.append(out)
        return ret
