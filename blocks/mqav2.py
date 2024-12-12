import torch
from functools import reduce
from torch import Tensor, nn
from typing import Tuple

class MultiQueryAttentionLayerV2(nn.Module):
  def __init__(self, input_channels:int, num_heads:int, key_dim:int, value_dim:int, dropout:float=0.0):
    super().__init__()
    self._num_heads = num_heads
    self._key_dim = key_dim
    self._value_dim = value_dim
    self._dropout = dropout
    self._channel_dim = input_channels

    self._query_proj = Tensor.glorot_uniform(self._num_heads, self._key_dim, self._channel_dim)
    self._key_proj = Tensor.glorot_uniform(self._channel_dim, self._key_dim)
    self._value_proj = Tensor.glorot_uniform(self._channel_dim, self._value_dim)
    self._output_proj = Tensor.glorot_uniform(self._channel_dim, self._num_heads, self._value_dim)

  def _reshape_input(self, t:Tensor) -> Tensor:
    s = t.shape
    num = reduce(lambda x,y:x*y, s[1:-1], 1)
    return t.view(s[0], num, s[-1])

  def __call__(self, inputs:Tuple[Tensor, Tensor]) -> Tensor:
    x, m = inputs

    reshaped_x = self._reshape_input(x)
    reshaped_m = self._reshape_input(m)

    q = torch.einsum('bnd,hkd->bnhk', reshaped_x, self._query_proj)
    k = torch.einsum('bmd,dk->bmk', reshaped_m, self._key_proj)
    logits = torch.einsum('bnhk,bmk->bnhm', q, k)

    logits = logits / Tensor([self._key_dim], dtype=x.dtype).sqrt()
    attention_scores = logits.softmax().dropout(self._dropout)

    v = torch.einsum('bmd,dv->bmv', reshaped_m, self._value_proj)
    o = torch.einsum('bnhm,bmv->bnhv', attention_scores, v)
    result = torch.einsum('bnhv,dhv->bnd', o, self._output_proj)

    return result.view(x.shape)
