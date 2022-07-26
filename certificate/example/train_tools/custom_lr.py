from torch.optim.lr_scheduler import StepLR
from math import log

__all__ = ["ConstantLR", "LinearLR", "LinearStepLR"]


class ConstantLR:
    def __init__(self, init_lr):
        self.init_lr = init_lr
        self.crnt_lr = self.init_lr

    def get_lr(self):
        return [self.crnt_lr]

    def get_last_lr(self):
        return [self.crnt_lr]

    def step(self):
        pass


class LinearLR:
    def __init__(self, init_lr, epoch, eta_min):
        self.init_lr = init_lr
        self.crnt_lr = init_lr

        tot_diff = init_lr - eta_min
        self.diff = tot_diff / (epoch - 1)

    def get_last_lr(self):
        return [self.crnt_lr]

    def step(self):
        self.crnt_lr -= self.diff


class LinearStepLR:
    def __init__(self, optimizer, init_lr, epoch, eta_min, decay_rate):
        n = int((log(eta_min) - log(init_lr)) / log(decay_rate)) + 1
        step_size = int(epoch / n)
        self.scheduler = StepLR(
            optimizer=optimizer, gamma=decay_rate, step_size=step_size
        )

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def step(self):
        self.scheduler.step()
