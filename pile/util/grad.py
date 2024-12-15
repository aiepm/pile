import torch

from torch import nn

def clip_gradients_global_norm(model:nn.Module, max_norm:float):
  grads = [p.grad for p in model.parameters() if p.grad is not None]
  
  global_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
  
  clip_coef = max_norm / (global_norm + 1e-6)
  
  if clip_coef < 1.0:
    for g in grads:
      g.mul_(clip_coef)
