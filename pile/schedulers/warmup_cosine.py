import math
import torch
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

class CosineAnnealingWarmRestartsWithDecay(_LRScheduler):
  def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, factor=1.0, last_epoch=-1):
    self.T_0 = T_0
    self.T_mult = T_mult
    self.eta_min = eta_min
    self.factor = factor
    self.base_max_lrs = [group['lr'] for group in optimizer.param_groups]
    self.T_i = T_0
    self.cycle = 0
    super(CosineAnnealingWarmRestartsWithDecay, self).__init__(optimizer, last_epoch)

  def get_lr(self):
    if self.last_epoch == 0 or self.last_epoch == self.T_i:
      # Adjust cycle
      if self.last_epoch > 0:
        self.cycle += 1
        self.T_i = self.T_i * self.T_mult
      self.base_max_lrs = [lr * (self.factor ** self.cycle) for lr in self.base_max_lrs]

    # Calculate learning rate using cosine annealing
    return [
      self.eta_min + (base_max_lr - self.eta_min) *
      (1 + math.cos(math.pi * (self.last_epoch % self.T_i) / self.T_i)) / 2
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
      'T_i': self.T_i,
      'cycle': self.cycle,
      'last_epoch': self.last_epoch,
      'base_max_lrs': self.base_max_lrs,
    }

  def load_state_dict(self, state_dict):
    """
    Restores the scheduler's state from the state dictionary.
    """
    self.T_0 = state_dict['T_0']
    self.T_mult = state_dict['T_mult']
    self.eta_min = state_dict['eta_min']
    self.factor = state_dict['factor']
    self.T_i = state_dict['T_i']
    self.cycle = state_dict['cycle']
    self.last_epoch = state_dict['last_epoch']
    self.base_max_lrs = state_dict['base_max_lrs']

class WarmupCosineScheduler:
  def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, T_0: int, T_mult: int = 2, factor: float = 0.5, eta_min: float = 1e-8):
    """
    A scheduler with a warmup phase followed by a cosine annealing schedule with decay.
    
    Args:
      optimizer (torch.optim.Optimizer): The optimizer to modify learning rates for.
      warmup_steps (int): Number of warmup steps.
      T_0 (int): Number of iterations for the first cosine annealing cycle.
      T_mult (int): Multiplicative factor for increasing cycle length.
      factor (float): Decay factor for max learning rate at each restart.
      eta_min (float): Minimum learning rate.
    """
    self.optimizer = optimizer
    self.warmup_epochs = warmup_steps
    self.epoch = 0
    
    self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, epoch / self.warmup_epochs))
    self.cosine_scheduler = CosineAnnealingWarmRestartsWithDecay(
      optimizer, T_0=T_0, T_mult=T_mult, factor=factor, eta_min=eta_min
    )

  def step(self):
    """
    Update the learning rate based on the current phase (warmup or cosine annealing).
    """
    if self.epoch < self.warmup_epochs:
      self.warmup_scheduler.step()
    else:
      self.cosine_scheduler.step()
    self.epoch += 1

  def get_lr(self):
    """
    Get the current learning rate from the optimizer.
    """
    if self.epoch < self.warmup_epochs:
      return self.warmup_scheduler.get_last_lr()
    return self.cosine_scheduler.get_last_lr()

  def state_dict(self):
    """
    Returns the state of the scheduler as a dictionary.
    """
    return {
      'epoch': self.epoch,
      'warmup_scheduler': self.warmup_scheduler.state_dict(),
      'cosine_scheduler': self.cosine_scheduler.state_dict()
    }

  def load_state_dict(self, state_dict):
    """
    Restores the scheduler's state from the state dictionary.
    """
    self.epoch = state_dict['epoch']
    self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
    self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler'])
