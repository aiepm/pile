import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

class WarmupCosineScheduler:
  def __init__(self, optimizer:torch.optim.Optimizer, num_epochs:int, warmup_epochs:int):
    self.num_epochs = num_epochs
    self.warmup_epochs = warmup_epochs
    self.train_epochs = num_epochs - warmup_epochs
    self.epoch = 0
    
    self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch : min(1.0, epoch/self.warmup_epochs))
    self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.train_epochs)

  def step(self):
    if self.epoch < self.warmup_epochs:
      self.warmup_scheduler.step()
    else:
      self.cosine_scheduler.step()
    self.epoch += 1
