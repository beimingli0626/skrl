from typing import Optional, Union

from packaging import version
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        min_lr: float = 0,
        last_epoch: int = -1,
        total_epochs: int = 100,
        verbose: bool = False,
    ) -> None:
        """Cosine scheduler

        .. note::

            This scheduler is only available for PPO at the moment.
            Applying it to other agents will not change the learning rate
        """
        if version.parse(torch.__version__) >= version.parse("2.2"):
            verbose = "deprecated"
        super().__init__(optimizer, last_epoch, verbose)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        self.total_epochs = total_epochs
        self.initial_lr = optimizer.param_groups[0]["lr"]
        self.min_lr = min_lr
        
    def step(self, epoch: Optional[int] = None) -> None:
        """
        Step scheduler
        """
        if epoch is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = self.min_lr + (self.initial_lr - self.min_lr) / 2 * (1 - math.cos(math.pi - math.pi * epoch / self.total_epochs))

            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
