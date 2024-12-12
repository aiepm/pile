import torch
from torch import Tensor, nn
from torchvision.ops import stochastic_depth
from functools import reduce
from .mqa_with_downsampling import MQAWithDownsampling
from .mqav2 import MultiQueryAttentionLayerV2
from .mnv4_layer_scale import MNV4LayerScale

class MultiHeadAttention(nn.Module):
  def __init__(self, input_channels:int, num_heads:int, key_dim:int, value_dim:int, dropout_rate:float=0.0):
    super().__init__()
    self._key_dim = key_dim
    self._value_dim = value_dim

    self._query_proj = nn.Parameter(torch.zeros(num_heads, self._key_dim, input_channels))
    self._key_proj = nn.Parameter(torch.zeros(num_heads, self._key_dim, input_channels))
    self._value_proj = nn.Parameter(torch.zeros(num_heads, self._value_dim, input_channels))
    score_normalization = lambda x: x / Tensor([self._key_dim], dtype=x.type).sqrt()
    self._attention_score = lambda sim: nn.Sequential(
        nn.Softmax(),
        nn.Dropout(dropout_rate)
    )(score_normalization(sim))
    self._output_proj = nn.Parameter(torch.zeros(input_channels, num_heads, self._value_dim))

  def _reshape_input(self, t:Tensor) -> Tensor:
    num = reduce(lambda x,y:x*y, t.shape[1:-1], 1)
    return t.view(t.shape[0], num, t.shape[-1])

  def forward(self, x:Tensor) -> Tensor:
    rx = self._reshape_input(x)

    q = torch.einsum('bnd,hkd->bnhk', rx, self._query_proj)
    k = torch.einsum('bnd,hkd->bnhk', rx, self._key_proj)
    v = torch.einsum('bnd,hvd->bnhv', rx, self._value_proj)
    sim = torch.einsum('bnkh,bmhk->bnhm', q, k)
    attention_scores = self._attention_score(sim)
    o = torch.einsum('bnhm,bmhv->bnhv', attention_scores, v)
    output = torch.einsum('bnhv,dhv->bnd', o, self._output_proj)
    
    return output

class MHSA(nn.Module):
  def __init__(
      self,
      input_dim,
      num_heads=8,
      key_dim=64,
      value_dim=64,
      use_multi_query=False,
      query_h_strides=1,
      query_w_strides=1,
      kv_strides=1,
      downsampling_dw_kernel_size=3,
      dropout=0.0,
      use_bias=False,
      use_cpe=False,
      cpe_dw_kernel_size=7,
      stochastic_depth_drop_rate=None,
      use_residual=True,
      use_layer_scale=True,
      layer_scale_init_value=1e-5,
      norm_momentum=0.99,
      norm_epsilon=0.001,
      output_intermediate_endpoints=False,
      **kwargs,
  ):
    super().__init__()
    self._key_dim = key_dim
    self._value_dim = value_dim
    self._use_multi_query = use_multi_query
    self._query_w_strides = query_w_strides
    self._downsampling_dw_kernel_size = downsampling_dw_kernel_size
    self._use_bias = use_bias
    self._use_residual = use_residual
    self._use_layer_scale = use_layer_scale
    self._layer_scale_init_value = layer_scale_init_value
    self._output_intermediate_endpoints = output_intermediate_endpoints

    self._input_norm = nn.BatchNorm2d(input_dim, eps=norm_epsilon, momentum=norm_momentum)

    self._cpe_dw_conv = (lambda x: nn.Conv2d(input_dim, input_dim, kernel_size=cpe_dw_kernel_size, groups=input_dim)(x) + x) if use_cpe else nn.Identity()

    num_heads = input_dim // self._key_dim if num_heads is None else num_heads

    self._attention = MultiHeadAttention(input_dim, num_heads, self._key_dim, self._value_dim, dropout)
    if self._use_multi_query:
      if query_h_strides > 1 or self._query_w_strides > 1 or kv_strides > 1:
        self._attention = MQAWithDownsampling(
            input_channels=input_dim,
            num_heads=num_heads,
            key_dim=self._key_dim,
            value_dim=self._value_dim,
            query_h_strides=query_h_strides,
            query_w_strides=self._query_w_strides,
            kv_strides=kv_strides,
            dw_kernel_size=self._downsampling_dw_kernel_size,
            dropout=dropout,
        )
      else:
        self._attention = MultiQueryAttentionLayerV2(
            input_channels=input_dim,
            num_heads=num_heads,
            key_dim=self._key_dim,
            value_dim=self._value_dim,
            dropout=dropout,
        )

    self._layer_scale = MNV4LayerScale(self._layer_scale_init_value, input_dim) if self._use_layer_scale else nn.Identity()
    self._stochastic_depth = lambda x: stochastic_depth(x, stochastic_depth_drop_rate, "row", self.training) if stochastic_depth_drop_rate else nn.Identity()

  def forward(self, x:Tensor) -> Tensor:
    cpe_outputs = self._cpe_dw_conv(x)
    shortcut = cpe_outputs
    x = self._input_norm(cpe_outputs)
    x = self._attention(x)
    x = self._layer_scale(x)

    if self._use_residual:
      x = self._stochastic_depth(x)
      x = x + shortcut

    if self._output_intermediate_endpoints:
      return x, {}

    return x
