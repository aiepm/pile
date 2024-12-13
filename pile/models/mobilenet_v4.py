from torch import Tensor, nn
from pile.blocks.mhsa import MHSA
from pile.blocks.universal_inverted_bottleneck import UniversalInvertedBottleneck
from pile.blocks.inverted_residual import InvertedResidual
from .specs import MODEL_SPECS

def convbn(in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, groups:int=1, bias:bool=False, norm:bool=True, act:bool=True):
  conv = nn.Sequential()
  padding = (kernel_size - 1) // 2
  conv.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups))
  if norm:
    conv.add_module('BatchNorm2d', nn.BatchNorm2d(out_channels))
  if act:
    conv.add_module('Activation', nn.ReLU6())
  return conv

def build_blocks(layer_spec):
  if not layer_spec.get('block_name'):
    return []
  block_names = layer_spec['block_name']
  layers = nn.Sequential()
  if block_names == "convbn":
    schema_ = ['in_channels', 'out_channels', 'kernel_size', 'stride']
    for i in range(layer_spec['num_blocks']):
      args = dict(zip(schema_, layer_spec['block_specs'][i]))
      block = convbn(**args)
      layers.add_module(f'convbn_{i}', block)
  elif block_names == "uib":
    schema_ =  ['in_channels', 'out_channels', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride', 'expand_ratio', 'mhsa']
    for i in range(layer_spec['num_blocks']):
      args = dict(zip(schema_, layer_spec['block_specs'][i]))
      mhsa = args.pop("mhsa") if "mhsa" in args else 0
      layers.add_module(f'uib_{i}', UniversalInvertedBottleneck(**args))
      if mhsa:
        mhsa_schema_ = [
            "input_dim", "num_heads", "key_dim", "value_dim", "query_h_strides", "query_w_strides", "kv_strides", 
            "use_layer_scale", "use_multi_query", "use_residual"
        ]
        args = dict(zip(mhsa_schema_, [args['out_channels']] + (mhsa)))
        layers.add_module(f'mhsa_{i}', MHSA(**args))
  elif block_names == "fused_ib":
    schema_ = ['in_channels', 'out_channels', 'stride', 'expand_ratio', 'activation']
    for i in range(layer_spec['num_blocks']):
      args = dict(zip(schema_, layer_spec['block_specs'][i]))
      layers.add_module(f'ires_{i}', InvertedResidual(**args))
  else:
    raise NotImplementedError
  return layers

class MobilenetV4(nn.Module):
  def __init__(self, sspec):
    super().__init__()
    self._spec = MODEL_SPECS[sspec]
    
    self._conv0 = build_blocks(self._spec['conv0'])
    self._layer1 = build_blocks(self._spec['layer1'])
    self._layer2 = build_blocks(self._spec['layer2'])
    self._layer3 = build_blocks(self._spec['layer3'])
    self._layer4 = build_blocks(self._spec['layer4'])
    self._layer5 = build_blocks(self._spec['layer5'])
    self._global_pooling = nn.AdaptiveAvgPool2d(1)

  def __call__(self, x:Tensor) -> Tensor:
    x0 = self._conv0(x)
    x1 = self._layer1(x0)
    x2 = self._layer2(x1)
    x3 = self._layer3(x2)
    x4 = self._layer4(x3)
    x5 = self._layer5(x4)
    x5 = self._global_pooling(x5)
    return [x1, x2, x3, x4, x5]
