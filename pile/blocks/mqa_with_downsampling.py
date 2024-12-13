import torch
from torch import nn, Tensor
from functools import reduce

class MQAWithDownsampling(nn.Module):
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
    super().__init__()
    self.num_heads = num_heads
    self.key_dim = key_dim
    self.query_h_strides = query_h_strides
    self.query_w_strides = query_w_strides

    self.head_dim = key_dim // num_heads

    query_strides = query_h_strides > 1 or query_w_strides > 1

    self._query_downsampling = nn.Sequential(
        nn.AvgPool2d(kernel_size=(query_h_strides, query_w_strides)),
        nn.BatchNorm2d(input_channels)
    ) if query_strides else nn.Identity()
    self._upsampling = nn.UpsamplingBilinear2d(size=(query_h_strides, query_w_strides)) if query_strides else nn.Identity()
    self._query_proj = nn.Conv2d(input_channels, num_heads*key_dim, kernel_size=1, stride=1, bias=False)

    self._key_dw_conv = nn.Sequential(
        nn.Conv2d(input_channels, input_channels, kernel_size=dw_kernel_size, stride=kv_strides, groups=input_channels),
        nn.BatchNorm2d(input_channels),
    ) if kv_strides > 1 else nn.Identity()

    self._value_dw_conv = nn.Sequential(
        nn.Conv2d(input_channels, input_channels, kernel_size=dw_kernel_size, stride=kv_strides, groups=input_channels),
        nn.BatchNorm2d(input_channels),
    ) if kv_strides > 1 else nn.Identity()

    self._key_proj = nn.Conv2d(input_channels, key_dim, kernel_size=1, stride=1, bias=False)
    self._value_proj = nn.Conv2d(input_channels, value_dim, kernel_size=1, stride=1, bias=False)
    score_normalization = lambda x: x / (self.key_dim ** 0.5)
    self._attention_score = lambda sim: nn.Sequential(
        nn.Softmax(dim=-1),
        nn.Dropout(dropout)
    )(score_normalization(sim))

    self._output_proj = nn.Conv2d(num_heads*key_dim, input_channels, kernel_size=1, stride=1, bias=False)

  def _reshape_projected_query(self, t:Tensor, num_heads:int, h_px:int, w_px:int, key_dim:int) -> Tensor:
    s = t.shape
    return t.view(s[0], num_heads, h_px * w_px, key_dim)

  def _reshape_input(self, t:Tensor) -> Tensor:
    s = t.shape
    return t.view(s[0], s[1], -1)

  def _reshape_output(self, t:Tensor, num_heads:int, h_px:int, w_px:int) -> Tensor:
    s = t.shape
    return t.view(s[0], s[-1] * num_heads, h_px, w_px)

  def _get_pixels(self, t:Tensor) -> int:
    return t.shape[2]

  def forward(self, x:Tensor) -> Tensor:
    px = self._get_pixels(x)
    q = self._query_downsampling(x)
    q = self._query_proj(q)
    q = self._reshape_projected_query(q, self.num_heads, px // self.query_h_strides, px // self.query_w_strides, self.key_dim)

    k = self._key_dw_conv(x)
    k = self._key_proj(k)
    k = self._reshape_input(k)

    sim = torch.einsum('bhlk,bkp->bhlp', q, k)
    attention_score = self._attention_score(sim)

    v = self._value_dw_conv(x)
    v = self._value_proj(v)
    v = self._reshape_input(v)
    o = torch.einsum('bhlp,bkp->bhlk', attention_score, v)
    o = self._reshape_output(o, self.num_heads, px // self.query_h_strides, px // self.query_w_strides)

    o = self._upsampling(o)
    o = self._output_proj(o)
    o = o.view(x.shape)
    return o
