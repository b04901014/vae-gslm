import math
from random import random
from functools import partial
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from utils.tensormask import TensorMask
from training_lib.losses import masked_l1_loss, masked_l2_loss

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class WeightStandardizedConv2d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def scaled_linear_beta_schedule(timesteps, hp):
    beta_start = hp.get("beta_start", 0.0015)
    beta_end = hp.get("beta_end", 0.0195)
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype = torch.float64) ** 2

def cosine_beta_schedule(timesteps, hp):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    s = hp.get("s", 0.008)
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        hp
    ):
        super().__init__()
        self.hp = hp
        timesteps = hp.timesteps
        p2_loss_weight_gamma = 0.
        p2_loss_weight_k = 1
        sampling_timesteps = hp.get("sampling_timesteps", None)
        loss_type = hp.get("loss_type", "l1")
        objective = hp.get("objective", "pred_noise")
        clamp_range = hp.get("clamp_range", [-1, 1])
        ddim_sampling_eta = hp.get("ddim_sampling_eta", 1.0)
        beta_schedule = hp.beta_schedule
        self.model = model
        self.objective = objective
        self.sigma = 1.0
        self.clamp_range = clamp_range

        if beta_schedule.identifier == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule.identifier == 'scaled_linear':
            betas = scaled_linear_beta_schedule(timesteps, hp.beta_schedule)
        elif beta_schedule.identifier == 'cosine':
            betas = cosine_beta_schedule(timesteps, hp.beta_schedule)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    @property
    def is_ddim_sampling(self):
        return self.sampling_timesteps < self.num_timesteps

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x: TensorMask, t: torch.tensor, cond: TensorMask, **kwargs):
        model_output = self.model(x, t, cond, **kwargs)
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x.value, t, pred_noise.value)
            x_start = TensorMask(x_start, model_output.mask).apply_mask()
        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x.value, t, x_start.value)
            pred_noise = TensorMask(pred_noise, model_output.mask).apply_mask()
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x: TensorMask, t: torch.tensor, cond: TensorMask, **kwargs):
        preds = self.model_predictions(x, t, cond, **kwargs)
        x_start = preds.pred_x_start.apply_mask().value
        #Clamp
        x_start.clamp_(self.clamp_range[0], self.clamp_range[1])
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x.value, t = t)
        model_mean = TensorMask(model_mean, preds.pred_x_start.mask)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x: TensorMask, t: int, cond: TensorMask, **kwargs):
        batched_times = torch.full((x.value.shape[0],), t, device = x.value.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, cond, **kwargs)
        noise = torch.randn_like(x.value) * self.sigma if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean.value + (0.5 * model_log_variance).exp() * noise
        pred_img = TensorMask(pred_img, model_mean.mask).apply_mask()
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, start, cond: TensorMask, **kwargs):
        img = start
        x_start = None
        sample_stride = self.num_timesteps // self.sampling_timesteps
        for t in reversed(range(0, self.num_timesteps, sample_stride)):
            img, x_start = self.p_sample(img, t, cond, **kwargs)
        return img

    @torch.no_grad()
    def ddim_sample(self, start, cond: TensorMask, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = start.value.shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = start

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond, **kwargs)
            x_start.value.clamp_(self.clamp_range[0], self.clamp_range[1])
            x_start = x_start.apply_mask()

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img.value) * self.sigma

            img = TensorMask(
                x_start.value * alpha_next.sqrt() + \
                c * pred_noise.value + \
                sigma * noise,
                x_start.mask
            ).apply_mask()

        return img

    @torch.no_grad()
    def sample(self, start, cond, **kwargs):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(start, cond, **kwargs)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return masked_l1_loss
        elif self.loss_type == 'l2':
            return masked_l2_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start: TensorMask,
                 t: torch.tensor,
                 cond: TensorMask,
                 **kwargs):
        loss_batch_weight = None
        if 'loss_batch_weight' in kwargs:
            loss_batch_weight = kwargs['loss_batch_weight']
            del kwargs['loss_batch_weight']
        b, c, n = x_start.value.shape
        noise = torch.randn_like(x_start.value)
        # noise sample
        x = self.q_sample(x_start=x_start.value, t=t, noise=noise)
        x = TensorMask(x, x_start.mask).apply_mask()

        model_out = self.model(x, t, cond, **kwargs)
        if self.objective == 'pred_noise':
            target = TensorMask(noise, x_start.mask).apply_mask()
        elif self.objective == 'pred_x0':
            target = x_start
        loss = self.loss_fn(model_out, target,
                            batch_weight=loss_batch_weight)
        return loss

    def forward(self, img: TensorMask,
                cond: TensorMask,
                **kwargs):
        b, device = img.value.size(0), img.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(img, t, cond, **kwargs)
