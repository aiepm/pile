import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmRestartsWithDecay(_LRScheduler):
  def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, factor=1.0, warmup_steps=0, last_epoch=-1):
    """
    Combines warmup and cosine annealing with restarts and decay.

    Args:
      optimizer (Optimizer): Wrapped optimizer.
      T_0 (int): Number of iterations for the first restart cycle.
      T_mult (int): Multiplicative factor for increasing cycle length.
      eta_min (float): Minimum learning rate.
      factor (float): Decay factor for max learning rate at each restart.
      warmup_steps (int): Number of steps for the warmup phase.
      last_epoch (int): The index of the last epoch. Default: -1.
    """
    self.T_0 = T_0
    self.T_mult = T_mult
    self.eta_min = eta_min
    self.factor = factor
    self.warmup_steps = warmup_steps
    self.cycle = 0
    self.T_i = T_0
    self.base_max_lrs = [group['lr'] for group in optimizer.param_groups]
    self.acc = 0
    super(CosineAnnealingWarmRestartsWithDecay, self).__init__(optimizer, last_epoch)

  def get_lr(self):
    if self.last_epoch < self.warmup_steps:
      # Linear warmup
      warmup_factor = (self.last_epoch + 1) / self.warmup_steps
      return [lr * warmup_factor for lr in self.base_max_lrs]

    if self.last_epoch == self.warmup_steps:
      self.acc += self.warmup_steps

    # Cosine annealing with decay
    current_epoch_in_cycle = self.last_epoch - self.acc
    cycle_length = self.T_i

    if current_epoch_in_cycle == cycle_length:
      # Start new cycle
      self.cycle += 1
      self.acc += self.T_i
      self.T_i = self.T_i * self.T_mult
      current_epoch_in_cycle = 0
      cycle_length = self.T_i
      self.base_max_lrs = [lr * self.factor for lr in self.base_max_lrs]

    return [
      self.eta_min + (base_max_lr - self.eta_min) *
      (1 + math.cos(math.pi * current_epoch_in_cycle / cycle_length)) / 2
      for base_max_lr in self.base_max_lrs
    ]

  def state_dict(self):
    """
    Returns the state of the scheduler as a dictionary.
    """
    return {
      'T_0': self.T_0,
      'T_mult': self.T_mult,
      'eta_min': self.eta_min,
      'factor': self.factor,
      'warmup_steps': self.warmup_steps,
      'cycle': self.cycle,
      'T_i': self.T_i,
      'last_epoch': self.last_epoch,
      'base_max_lrs': self.base_max_lrs,
      'acc': self.acc
    }

  def load_state_dict(self, state_dict):
    """
    Restores the scheduler's state from the state dictionary.
    """
    self.T_0 = state_dict['T_0']
    self.T_mult = state_dict['T_mult']
    self.eta_min = state_dict['eta_min']
    self.factor = state_dict['factor']
    self.warmup_steps = state_dict['warmup_steps']
    self.cycle = state_dict['cycle']
    self.T_i = state_dict['T_i']
    self.last_epoch = state_dict['last_epoch']
    self.base_max_lrs = state_dict['base_max_lrs']
    self.acc = state_dict['acc']

