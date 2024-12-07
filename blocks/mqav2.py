from functools import reduce
from tinygrad import Tensor
from typing import Tuple

class MultiQueryAttentionLayerV2:
  def __init__(self, input_channels:int, num_heads:int, key_dim:int, value_dim:int, dropout:float=0.0):
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

    q = Tensor.einsum('bnd,hkd->bnhk', reshaped_x, self._query_proj)
    k = Tensor.einsum('bmd,dk->bmk', reshaped_m, self._key_proj)
    logits = Tensor.einsum('bnhk,bmk->bnhm', q, k)

    logits = logits / Tensor([self._key_dim], dtype=x.dtype).sqrt()
    attention_scores = logits.softmax().dropout(self._dropout)

    v = Tensor.einsum('bmd,dv->bmv', reshaped_m, self._value_proj)
    o = Tensor.einsum('bnhm,bmv->bnhv', attention_scores, v)
    result = Tensor.einsum('bnhv,dhv->bnd', o, self._output_proj)

    return result.view(x.shape)
