from collections.abc import Iterable
from hparams.hp import Hparams
from torch import optim
from typing import Tuple
from torch.optim.lr_scheduler import LRScheduler


def getAdam(hp: Hparams,
            parameters: Iterable) -> optim.Optimizer:
    hp.check_arg_in_hparams('lr', 'beta1', 'beta2')
    return optim.Adam(parameters,
                      lr=hp.lr,
                      betas=(hp.beta1, hp.beta2),
                      eps=hp.get('eps', 1e-8),
                      weight_decay=hp.get('weight_decay', 0))


def getAdamW(hp: Hparams,
             parameters: Iterable) -> optim.Optimizer:
    hp.check_arg_in_hparams('lr', 'beta1', 'beta2')
    return optim.AdamW(parameters,
                       lr=hp.lr,
                       betas=(hp.beta1, hp.beta2),
                       eps=hp.get('eps', 1e-8),
                       weight_decay=hp.get('weight_decay', 0.01))


class ConstantLR(LRScheduler):
    def __init__(self, optimizer, lr, last_epoch=-1, verbose=False):
        self.lr = lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [self.lr
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [self.lr
                for base_lr in self.base_lrs]


# Can be added if we need more type of optimizers
# Now only stick with Adam
def optimizer_map(hp: Hparams,
                  parameters: Iterable) -> optim.Optimizer:
    hp.check_arg_in_hparams('identifier')
    if hp.identifier == 'Adam':
        return getAdam(hp, parameters)
    if hp.identifier == 'AdamW':
        return getAdamW(hp, parameters)
    raise NotImplementedError("The specified optimizer"
                              f"{hp.optim.optimizer} is not implemented yet.")


# Can be added if we need more type of learning rate schedules
def scheduler_map(hp: Hparams,
                  optimizer: optim.Optimizer,
                  total_steps: int) -> Tuple[LRScheduler, str]:
    hp.check_arg_in_hparams('identifier')
    scheduler_stack, milestones, milestone = [], [], 0
    if hp.has('warmup_steps'):
        def warmup_lr(current_step: int) -> float:
            return float(current_step) / float(max(1, hp.warmup_steps))
        scheduler_stack.append(optim.lr_scheduler.LambdaLR(optimizer,
                                                           warmup_lr))
        milestone += hp.warmup_steps
        milestones.append(milestone)
    if hp.has('flat_steps'):
        scheduler_stack.append(
            optim.lr_scheduler.LambdaLR(optimizer,
                                        lambda x: 1.0))
        milestone += hp.flat_steps
        milestones.append(milestone)
    assert total_steps > milestone
    total_steps = total_steps - milestone - hp.get("finish_steps", 0)
    if hp.identifier in ['linear_decay', 'triangle']:
        def linear_lr(current_step: int) -> float:
            return max(
                0.0, (float(total_steps - current_step) /
                      float(total_steps)
                      )
            )
        scheduler_stack.append(
            optim.lr_scheduler.LambdaLR(optimizer, linear_lr))
    elif hp.identifier == 'constant':
        scheduler_stack.append(
            optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.0))
    elif hp.identifier == 'cosine':
        scheduler_stack.append(
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=hp.get('min_lr', 0))
        )
    else:
        raise NotImplementedError
    if hp.has('finish_steps'):
        assert hp.get('min_lr', 0) != 0
        scheduler_stack.append(
            ConstantLR(optimizer, hp.min_lr))
        milestone += total_steps
        milestones.append(milestone)
    if len(scheduler_stack) > 1:
        return optim.lr_scheduler.SequentialLR(
            optimizer, scheduler_stack, milestones
        ), 'step'
    return scheduler_stack[0], 'step'


def create_optimizer(hp: Hparams,
                     parameters: Iterable,
                     total_steps: int
                     ) -> Tuple[optim.Optimizer, LRScheduler]:
    hp.check_arg_in_hparams("optimizer", "scheduler")
    if hp.optimizer.get("exclude_norm_and_bias_from_weight_decay", False):
        # Exclude norm and biases from weight decay
        parameters = list(parameters)
        onedim_params = [p for p in parameters if p.ndim == 1]
        other_params = [p for p in parameters if p.ndim != 1]
        parameters = [
            {'params': other_params},
            {'params': onedim_params, 'weight_decay': 0}
        ]
    optimizer = optimizer_map(hp.optimizer, parameters)
    scheduler, interval = scheduler_map(hp.scheduler, optimizer, total_steps)
    scheduler = {
        'scheduler': scheduler,
        'interval': interval
    }
    return optimizer, scheduler
