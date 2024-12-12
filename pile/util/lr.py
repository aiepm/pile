import torch

def get_current_lr(optimizer:torch.optim.Optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']
