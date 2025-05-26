import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tensormask import TensorMask
from typing import Callable, Optional
from hparams.hp import Hparams


def masked_loss(x: TensorMask,
                y: TensorMask,
                fn: Callable,
                time_reduction: bool = False,
                batch_reduction: bool = False,
                batch_weight: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
    a = x.flatten().apply_mask().value
    b = y.flatten().apply_mask().value
    out = fn(a, b).mean(-1).sum(-1)
    if batch_weight is not None:
        out = out * batch_weight
    if time_reduction and batch_reduction:
        return out.sum() / x.length.sum()
    if time_reduction:
        return (out / x.length).mean()
    if batch_reduction:
        return out.mean()
    return out.sum()


def cross_entropy_loss(a, b, reduction):
    return F.cross_entropy(a, b, reduction=reduction, ignore_index=-100)


def masked_ce_loss(x: TensorMask,
                   y: TensorMask,
                   reduction: str = "sum"
                   ) -> torch.Tensor:
    a = x.apply_mask().value.reshape([-1, x.size(-1)])
    b = y.apply_mask(-100).value.reshape([-1])
    out = cross_entropy_loss(a, b, reduction=reduction)
    return out


def l1_loss(a, b):
    return torch.abs(a - b)


def masked_l1_loss(x: TensorMask,
                   y: TensorMask,
                   time_reduction: bool = False,
                   batch_reduction: bool = False,
                   batch_weight: Optional[torch.Tensor] = None
                   ) -> torch.Tensor:
    return masked_loss(x, y, fn=l1_loss,
                       time_reduction=time_reduction,
                       batch_reduction=batch_reduction,
                       batch_weight=batch_weight)


def l2_loss(a, b):
    return torch.pow(a - b, 2)


def masked_l2_loss(x: TensorMask,
                   y: TensorMask,
                   time_reduction: bool = False,
                   batch_reduction: bool = False,
                   batch_weight: Optional[torch.Tensor] = None
                   ) -> torch.Tensor:
    return masked_loss(x, y, fn=l2_loss,
                       time_reduction=time_reduction,
                       batch_reduction=batch_reduction,
                       batch_weight=batch_weight)


class InfoNCE(nn.Module):
    def __init__(self, hp: Hparams, dim1: int, dim2: int):
        super().__init__()
        hp.check_arg_in_hparams("dim", "num_negatives")
        self.max_neg = hp.num_negatives
        self.middle_dim = hp.dim
        self.linear1 = nn.Linear(dim1, self.middle_dim)
        self.linear2 = nn.Linear(dim2, self.middle_dim)
        self.hp = hp

    def forward(self,
                q: TensorMask,
                p: TensorMask) -> torch.Tensor:
        """q, p: B, T, C, statically compilable version w.o. warning
        to be implemented
        """
        mask = q.mask
        b, t, c = q.size()
        q, p = q.value[mask], p.value[mask]  # BT, C
#        q, p = F.normalize(q, dim=-1), F.normalize(p, dim=-1)
        if self.max_neg is not None:
            indices = torch.randperm(q.size(0),
                                     dtype=torch.long, device=q.device)
            indices = indices[:self.max_neg]
            q, p = q[indices], p[indices]
        logits = self.linear1(q) @ self.linear2(p).T  # BT, BT
        logits = logits / self.middle_dim ** 0.5
        labels = torch.arange(logits.size(0), device=q.device)
        loss = F.cross_entropy(logits, labels, reduction='sum')
        return loss

    def band_mask(self, x: torch.Tensor) -> torch.Tensor:
        """x: BT, BT"""
        band_size = self.hp.get("band_size", 1)
        if band_size == 1:
            return x
        one = torch.ones_like(x)
        u_mask = torch.triu(one, diagonal=band_size)
        d_mask = torch.tril(one, diagonal=-band_size)
        mask = (u_mask + d_mask).bool()
        mask = mask.fill_diagonal_(True)
        ret = torch.where(mask, x, float('-inf'))
        return ret

    def sum_probs_topk_off_diagonal(self, probs: torch.Tensor) -> torch.Tensor:
        n = probs.size(0)
        probs = probs.flatten()[1:].view(n-1, n+1)[:, :-1].reshape(n, n-1)
        topk = torch.topk(probs, self.hp.softCLR.topk)
        return topk.values.sum(-1)

    @torch.no_grad()
    def self_cosine_similarity(self,
                               x: torch.tensor,
                               mask: torch.tensor
                               ) -> torch.Tensor:
        """x: BT, C, mask: BT,"""
        norm_x = torch.clamp((x ** 2).sum(-1, keepdim=True),
                             min=self.hp.softCLR.get("epsilon", 1e-8))
        norm_x = x / norm_x
        cosine_similarity = norm_x @ norm_x.T  # BT, BT
        cosine_similarity = torch.where(mask[None],
                                        cosine_similarity, float('-inf'))
        topk = torch.topk(cosine_similarity, self.hp.softCLR.topk)
        probs = torch.softmax(topk.values, -1)
        labels = torch.zeros_like(cosine_similarity).to(probs.dtype)
        labels.scatter_(1, topk.indices, probs)
        return labels.detach()


class CPC(nn.Module):
    def __init__(self, hp: Hparams, dim1: int, dim2: int):
        super().__init__()
        hp.check_arg_in_hparams("num_predictors",
                                "num_negatives",
                                "dim")
        self.max_neg = hp.num_negatives
        self.max_neg_same = hp.get("num_neg_same_utterance", 0)
        self.num_predictors = hp.num_predictors
        self.middle_dim = hp.dim
        # Note that we don't use rpe embedding as we assume working
        # on transformer layer outputs
        self.predictors = nn.ModuleList([
            nn.Linear(dim1, self.middle_dim)
            for _ in range(self.num_predictors)
        ])
        self.hp = hp
        self.linearp = nn.ModuleList([
            nn.Linear(dim2, self.middle_dim)
            for _ in range(self.num_predictors)
        ])

    def forward(self,
                q: TensorMask,
                p: TensorMask) -> torch.Tensor:
        """q, p: B, T, C, statically compilable version w.o. warning
        to be implemented
        """
        losses = 0
        iter_p, iter_q = p, q
        for k in range(self.num_predictors):
            # shift by one frame (B, T-k-1, ...)
            if k == 0:
                p, q = iter_p, iter_q
            else:
                p, q = iter_p.pop_left(k), iter_q.pop(k)
            _q = q
            mask = q.mask
            b, t, c = q.size()
            q, p = q.value[mask], p.value[mask]  # BT, C
            q = self.predictors[k](q)
            p = self.linearp[k](p)
            _p = p
            #  Sometimes it may sample positive instances, but low prob.
            neg_indicies = torch.randint(low=0, high=q.size(0),
                                         size=(q.size(0), self.max_neg),
                                         device=q.device)
            neg = p[neg_indicies]
            p = torch.cat([p[:, None], neg], 1)  # BT, M+1, C
            #  Produce negative examples from the same utterance
            if self.max_neg_same > 0:
                length = _q.length
                s_neg_indicies = torch.randint(2**63 - 1,
                                               size=(b, t, self.max_neg_same),
                                               device=q.device)
                s_neg_indicies = s_neg_indicies % length[:, None, None]
                cumsum_length = torch.cumsum(length, dim=0)
                dummy = torch.zeros([1], device=cumsum_length.device,
                                    dtype=cumsum_length.dtype)
                cumsum_length = torch.cat([dummy, cumsum_length], 0)
                cumsum_length = cumsum_length[: -1]
                s_neg_indicies = s_neg_indicies + cumsum_length[:, None, None]
                s_neg_indicies = s_neg_indicies[mask]
                p = torch.cat([p, _p[s_neg_indicies]], 1)
            logits = q[:, None] @ p.transpose(-1, -2)
            logits = logits[:, 0] / self.middle_dim ** 0.5
            labels = torch.zeros(logits.size(0),
                                 device=q.device, dtype=torch.long)
            loss = F.cross_entropy(logits, labels, reduction='sum')
            losses += loss
        return losses


def eos_loss(logits: TensorMask):
    """logits: B, T"""
    mask = logits.mask
    labels = torch.zeros_like(logits.value)
    labels.scatter_(1, (logits.length - 1)[..., None], 1.0)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.value, labels,
        pos_weight=torch.tensor(25.0, device=logits.value.device,
                                dtype=torch.float32),
        reduction='none'
    )
    loss = torch.where(mask, loss, 0.0).sum()
    return loss
