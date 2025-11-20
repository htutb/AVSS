import math

from torch.optim.lr_scheduler import _LRScheduler


class TunableNoamLR(_LRScheduler):
    """
    Tunable Noam Learning Rate Scheduler

    Args:
        optimizer (torch.optim.Optimizer): optimizer
        model_size (int): model size (d_model)
        warmup_steps (int): number of warmup steps
        k1 (int): scale parameter in lr during warmup
        k2 (int): scale parameter in lr after warmup
        min_lr (float): minimum learning rate after decay
        last_step (int): index of the last step
        current_epoch (int): current epoch
    """

    def __init__(
        self,
        optimizer,
        model_size,
        k1,
        k2,
        warmup_steps=4000,
        total_num_steps=40000,
        min_lr=0.0,
        last_step=-1,
    ):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.k1 = k1
        self.k2 = k2
        self.total_num_steps = total_num_steps
        super().__init__(optimizer=optimizer, last_epoch=last_step)

    def get_lr(self):
        self.last_step = self.last_epoch
        step = max(1, self.last_step + 1)
        current_epoch = self.total_num_steps // step
        if step <= self.warmup_steps:
            lr = (
                self.k1 * self.model_size ** (-0.5) * step * self.warmup_steps ** (-1.5)
            )
        else:
            lr = self.k2 * 0.98 ** (current_epoch // 2)

        lr = max(lr, self.min_lr)
        return [lr for _ in self.base_lrs]
