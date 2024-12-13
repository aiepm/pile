import torch
from torch import Tensor, nn

class MNV4LayerScale(nn.Module):
  def __init__(self, init_value:float, embedding_dim:int):
    super().__init__()
    self._init_value = init_value
    self._embedding_dim = embedding_dim
    self._gamma = nn.Parameter(self._init_value * torch.ones(self._embedding_dim,), requires_grad=True)

  def forward(self, x:Tensor) -> Tensor:
    return x * self._gamma.to(dtype=x.dtype).view(1, -1, 1, 1)

