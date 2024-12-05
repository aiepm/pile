from tinygrad import nn, Tensor
from functools import reduce

class MQAWithDownsampling:
  def __init__(self,
               input_channels:int,
               num_heads:int,
               key_dim:int,
               value_dim:int,
               query_h_strides:int,
               query_w_strides:int,
               kv_strides:int,
               dw_kernel_size:int=3,
               dropout:float=0.0):
    self.num_heads = num_heads
    self.key_dim = key_dim
    self.value_dim = value_dim
    self.query_h_strides = query_h_strides
    self.query_w_strides = query_w_strides
    self.kv_strides = kv_strides
    self.dw_kernel_size = dw_kernel_size
    self.dropout = dropout

    self.head_dim = key_dim // num_heads

    if self.query_h_strides > 1 or self.query_w_strides > 1:
      self._query_downsampling_norm = nn.BatchNorm2d(input_channels)
      self._upsampling = nn.ConvTranspose2d(3, 3, kernel_size=3, stride=(self.query_h_strides, self.query_w_strides), padding=1, output_padding=1)
    self._query_proj = nn.Conv2d(input_channels, num_heads*key_dim, kernel_size=1, stride=1, bias=False)

    if self.kv_strides > 1:
      self._key_dw_conv = [
          nn.Conv2d(input_channels, input_channels, kernel_size=dw_kernel_size, stride=kv_strides, groups=input_channels),
          nn.BatchNorm2d(input_channels)
      ]
      self._value_dw_conv = [
          nn.Conv2d(input_channels, input_channels, kernel_size=dw_kernel_size, stride=kv_strides, groups=input_channels),
          nn.BatchNorm2d(input_channels)
      ]
    self._key_proj = nn.Conv2d(input_channels, key_dim, kernel_size=1, stride=1, bias=False)
    self._value_proj = nn.Conv2d(input_channels, value_dim, kernel_size=1, stride=1, bias=False)

    self._output_proj = nn.Conv2d(num_heads*key_dim, input_channels, kernel_size=1, stride=1, bias=False)

  def _reshape_projected_query(self, t:Tensor, num_heads:int, h_px:int, w_px:int, key_dim:int) -> Tensor:
    s = t.shape
    return t.view(s[0], h_px * w_px, num_heads, key_dim)

  def _reshape_input(self, t:Tensor) -> Tensor:
    s = t.shape
    num = reduce(lambda x, y: x * y, s[1:-1], 1)
    return t.view(s[0], num, s[-1])

  def _get_pixels(self, t:Tensor) -> int:
    return t.shape[1]

  def __call__(self, inputs:Tensor) -> Tensor:
    x = inputs
    px = self._get_pixels(x)

    q = x
    if self.query_h_strides > 1 or self.query_w_strides > 1:
      q = q.avg_pool2d(kernel_size=(self.query_w_strides, self.query_w_strides))
      q = self._query_downsampling_norm(q)
    q = self._query_proj(q)
    q = self._reshape_projected_query(q, self.num_heads, px // self.query_h_strides, px // self.query_w_strides, self.key_dim)

    k = x
    if self.kv_strides > 1:
      k = k.sequential(self._key_dw_conv)
    k = self._key_proj(k)
    k = self._reshape_input(k)

    logits = Tensor.einsum('blhk,bpk->blhp', q, k)

    logits = logits / Tensor([self._key_dim], dtype=x.dtype).sqrt()
    attention_score = logits.softmax().dropout(self.dropout)

    v = x
    if self.kv_strides > 1:
      v = v.sequential(self._value_dw_conv)
    v = self._value_proj(v)
    v = self._reshape_input(v)
    o = Tensor.einsum('blhp,bpk->blhk', attention_score, v)
    o = self._reshape_output(o, self.num_heads, px // self.query_h_strides, px // self.query_w_strides)

    if self.query_w_strides > 1 or self.query_w_strides > 1:
      o = self._upsampling(o)

    o = o.view(x.shape)
    return o
