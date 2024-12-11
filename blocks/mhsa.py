from tinygrad import Tensor, nn
from functools import reduce
from mqa_with_downsampling import MQAWithDownsampling
from mqav2 import MultiQueryAttentionLayerV2
from mnv4_layer_scale import MNV4LayerScale
from stochastic_dropout import StochasticDepth

class MultiHeadAttention:
  def __init__(self, input_channels:int, num_heads:int, key_dim:int, value_dim:int, dropout_rate:float=0.0):
    self._key_dim = key_dim
    self._value_dim = value_dim
    self._dropout_rate = dropout_rate

    self._query_proj = Tensor.glorot_uniform(num_heads, self._key_dim, input_channels)
    self._key_proj = Tensor.glorot_uniform(num_heads, self._key_dim, input_channels)
    self._value_proj = Tensor.glorot_uniform(num_heads, self._value_dim, input_channels)
    self._score_normalization = lambda x: x / Tensor([self._key_dim], dtype=x.type).sqrt()
    self._output_proj = Tensor.glorot_uniform(input_channels, num_heads, self._value_dim)

  def _reshape_input(self, t:Tensor) -> Tensor:
    num = reduce(lambda x,y:x*y, t.shape[1:-1], 1)
    return t.view(t.shape[0], num, t.shape[-1])

  def __call__(self, x:Tensor) -> Tensor:
    rx = self._reshape_input(x)

    q = Tensor.einsum('bnd,hkd->bnhk', rx, self._query_proj)
    k = Tensor.einsum('bnd,hkd->bnhk', rx, self._key_proj)
    v = Tensor.einsum('bnd,hvd->bnhv', rx, self._value_proj)
    attention_scores = self._score_normalization(Tensor.einsum('bnkh,bmhk->bnhm', q, k)).softmax().dropout(self._dropout_rate)
    o = Tensor.einsum('bnhm,bmhv->bnhv', attention_scores, v)
    output = Tensor.einsum('bnhv,dhv->bnd', o, self._output_proj)
    
    return output

class MHSA:
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
    self._key_dim = key_dim
    self._value_dim = value_dim
    self._use_multi_query = use_multi_query
    self._query_h_strides = query_h_strides
    self._query_w_strides = query_w_strides
    self._kv_strides = kv_strides
    self._downsampling_dw_kernel_size = downsampling_dw_kernel_size
    self._dropout = dropout
    self._use_bias = use_bias
    self._use_cpe = use_cpe
    self._cpe_dw_kernel_size = cpe_dw_kernel_size
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._use_residual = use_residual
    self._use_layer_scale = use_layer_scale
    self._layer_scale_init_value = layer_scale_init_value
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._output_intermediate_endpoints = output_intermediate_endpoints

    self._input_norm = nn.BatchNorm2d(input_dim, eps=self._norm_epsilon, momentum=self._norm_momentum)

    if self._use_cpe:
      self._cpe_dw_conv = nn.Conv2d(input_dim, input_dim, kernel_size=self._cpe_dw_kernel_size, groups=input_dim)

    if num_heads is None:
      num_heads = input_dim // self._key_dim
    else:
      num_heads = num_heads

    if self._use_multi_query:
      if self._query_h_strides > 1 or self._query_w_strides > 1 or self._kv_strides > 1:
        self._multi_query_attention = MQAWithDownsampling(
            input_channels=input_dim,
            num_heads=num_heads,
            key_dim=self._key_dim,
            value_dim=self._value_dim,
            query_h_strides=self._query_h_strides,
            query_w_strides=self._query_w_strides,
            kv_strides=self._kv_strides,
            dw_kernel_size=self._downsampling_dw_kernel_size,
            dropout=self._dropout,
        )
      else:
        self._multi_query_attention = MultiQueryAttentionLayerV2(
            input_channels=input_dim,
            num_heads=num_heads,
            key_dim=self._key_dim,
            value_dim=self._value_dim,
            dropout=self._dropout,
        )
    else:
      self._multihead_attention = MultiHeadAttention(input_dim, num_heads, self._key_dim, self._value_dim, self._dropout)

    if self._use_layer_scale:
      self._layer_scale = MNV4LayerScale(self._layer_scale_init_value, input_dim)

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = StochasticDepth(self._stochastic_depth_drop_rate)

  def __call__(self, x:Tensor) -> Tensor:
    ox = x
    cpe_outputs = x
    if self._use_cpe:
      x = self._cpe_dw_conv(x)
      x = x + ox
      cpe_outputs = x

    shortcut = cpe_outputs
    x = self._input_norm(cpe_outputs)

    if self._use_multi_query:
      if (
          self._query_h_strides > 1
          or self._query_w_strides > 1
          or self._kv_strides > 1
      ):
        x = self._multi_query_attention(x)
      else:
        x = self._multi_query_attention((x, x))
    else:
      x = self._multi_head_attention(x)

    if self._use_layer_scale:
      x = self._layer_scale(x)

    if self._use_residual:
      if self._stochastic_depth:
        x = self._stochastic_depth(x)
      x = x + shortcut

    if self._output_intermediate_endpoints:
      return x, {}

    return x
