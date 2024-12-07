from tinygrad import Tensor, nn

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
    self._input_dim = input_dim
    self._num_heads = num_heads
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

    self._input_norm = nn.BatchNorm2d(self._input_dim, eps=self._norm_epsilon, momentum=self._norm_momentum)

    if self._use_cpe:
      self._cpe_dw_conv = nn.Conv2d(self._input_dim, self._input_dim, kernel_size=self._cpe_dw_kernel_size, groups=self._input_dim)

    if self._num_heads is None:
      num_heads = self._input_dim // self._key_dim
    else:
      num_heads = self._num_heads


  def __call__(self, x:Tensor) -> Tensor:
    resx = x

