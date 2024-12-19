from torch import Tensor, nn
from typing import List
from pile.blocks import MHSA
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


class CustomSmall(nn.Module):
  def __init__(self):
    super().__init__()
    self._block0 = convbn(in_channels=3, out_channels=32, kernel_size=3, stride=2)
    self._block1 = InvertedResidual(in_channels=32, out_channels=64, kernel_size=3, stride=2, expand_ratio=4.0, activation=True)
    self._block2 = InvertedResidual(in_channels=64, out_channels=128, kernel_size=3, stride=2, expand_ratio=4.0, activation=True)
    self._block3 = InvertedResidual(in_channels=128, out_channels=256, kernel_size=3, stride=1, expand_ratio=4.0, activation=True)
    self._block4 = InvertedResidual(in_channels=256, out_channels=256, kernel_size=5, stride=1, expand_ratio=4.0, activation=True)
    self._block5 = InvertedResidual(in_channels=256, out_channels=256, kernel_size=3, stride=1, expand_ratio=4.0, activation=True, squeeze_excite=True)
    self._block6 = InvertedResidual(in_channels=256, out_channels=256, kernel_size=5, stride=1, expand_ratio=4.0, activation=True, squeeze_excite=True)
    self._block7 = InvertedResidual(in_channels=256, out_channels=512, kernel_size=3, stride=2, expand_ratio=4.0, activation=True)
    self._block8 = InvertedResidual(in_channels=512, out_channels=512, kernel_size=3, stride=1, expand_ratio=4.0, activation=True, squeeze_excite=True)
    self._block9 = InvertedResidual(in_channels=512, out_channels=512, kernel_size=5, stride=1, expand_ratio=4.0, activation=True, squeeze_excite=True)

    self._block10 = nn.Sequential(
        convbn(in_channels=512, out_channels=960, kernel_size=1, stride=1),
        convbn(in_channels=960, out_channels=1280, kernel_size=1, stride=1),
        nn.AdaptiveAvgPool2d(1)
    )

    self._global_pooling = nn.AdaptiveAvgPool2d(1)

  def forward(self, x:Tensor) -> Tensor:
    x0 = self._block0(x)
    x1 = self._block1(x0)
    x2 = self._block2(x1)
    x3 = self._block3(x2)
    x4 = self._block4(x3)
    x5 = self._block5(x4)
    x6 = self._block6(x5)
    x7 = self._block7(x6)
    x8 = self._block8(x7)
    x9 = self._block9(x8)
    x10 = self._block10(x9)
    x10 = self._global_pooling(x10)
    return x10


class MobilenetV4ConvLarge(nn.Module):
  def __init__(self):
    super().__init__()
    self._block0 = convbn(in_channels=3, out_channels=24, kernel_size=3, stride=2)
    
    self._block1 = InvertedResidual(in_channels=24, out_channels=48, stride=2, expand_ratio=4.0, activation=True)
    
    self._block2 = nn.Sequential(
        UniversalInvertedBottleneck(in_channels=48, out_channels=96, start_dw_kernel_size=3, middle_dw_kernel_size=5, middle_dw_downsample=True, stride=2, expand_ratio=4.0),
        UniversalInvertedBottleneck(in_channels=96, out_channels=96, start_dw_kernel_size=3, middle_dw_kernel_size=3, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
    )

    self._block3 = nn.Sequential(
        UniversalInvertedBottleneck(in_channels=96, out_channels=192, start_dw_kernel_size=3, middle_dw_kernel_size=5, middle_dw_downsample=True, stride=2, expand_ratio=4.0),

        UniversalInvertedBottleneck(in_channels=192, out_channels=192, start_dw_kernel_size=3, middle_dw_kernel_size=3, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
        UniversalInvertedBottleneck(in_channels=192, out_channels=192, start_dw_kernel_size=3, middle_dw_kernel_size=3, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
        UniversalInvertedBottleneck(in_channels=192, out_channels=192, start_dw_kernel_size=3, middle_dw_kernel_size=3, middle_dw_downsample=True, stride=1, expand_ratio=4.0),

        UniversalInvertedBottleneck(in_channels=192, out_channels=192, start_dw_kernel_size=3, middle_dw_kernel_size=5, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
        
        UniversalInvertedBottleneck(in_channels=192, out_channels=192, start_dw_kernel_size=5, middle_dw_kernel_size=3, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
        UniversalInvertedBottleneck(in_channels=192, out_channels=192, start_dw_kernel_size=5, middle_dw_kernel_size=3, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
        UniversalInvertedBottleneck(in_channels=192, out_channels=192, start_dw_kernel_size=5, middle_dw_kernel_size=3, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
        UniversalInvertedBottleneck(in_channels=192, out_channels=192, start_dw_kernel_size=5, middle_dw_kernel_size=3, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
        UniversalInvertedBottleneck(in_channels=192, out_channels=192, start_dw_kernel_size=5, middle_dw_kernel_size=3, middle_dw_downsample=True, stride=1, expand_ratio=4.0),

        UniversalInvertedBottleneck(in_channels=192, out_channels=192, start_dw_kernel_size=3, middle_dw_kernel_size=0, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
    )

    self._block4 = nn.Sequential(
        UniversalInvertedBottleneck(in_channels=192, out_channels=512, start_dw_kernel_size=5, middle_dw_kernel_size=5, middle_dw_downsample=True, stride=2, expand_ratio=4.0),

        UniversalInvertedBottleneck(in_channels=512, out_channels=512, start_dw_kernel_size=5, middle_dw_kernel_size=5, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
        UniversalInvertedBottleneck(in_channels=512, out_channels=512, start_dw_kernel_size=5, middle_dw_kernel_size=5, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
        UniversalInvertedBottleneck(in_channels=512, out_channels=512, start_dw_kernel_size=5, middle_dw_kernel_size=5, middle_dw_downsample=True, stride=1, expand_ratio=4.0),

        UniversalInvertedBottleneck(in_channels=512, out_channels=512, start_dw_kernel_size=5, middle_dw_kernel_size=0, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
        
        UniversalInvertedBottleneck(in_channels=512, out_channels=512, start_dw_kernel_size=5, middle_dw_kernel_size=3, middle_dw_downsample=True, stride=1, expand_ratio=4.0),

        UniversalInvertedBottleneck(in_channels=512, out_channels=512, start_dw_kernel_size=5, middle_dw_kernel_size=0, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
        UniversalInvertedBottleneck(in_channels=512, out_channels=512, start_dw_kernel_size=5, middle_dw_kernel_size=0, middle_dw_downsample=True, stride=1, expand_ratio=4.0),


        UniversalInvertedBottleneck(in_channels=512, out_channels=512, start_dw_kernel_size=5, middle_dw_kernel_size=3, middle_dw_downsample=True, stride=1, expand_ratio=4.0),

        UniversalInvertedBottleneck(in_channels=512, out_channels=512, start_dw_kernel_size=5, middle_dw_kernel_size=5, middle_dw_downsample=True, stride=1, expand_ratio=4.0),

        UniversalInvertedBottleneck(in_channels=512, out_channels=512, start_dw_kernel_size=5, middle_dw_kernel_size=0, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
        UniversalInvertedBottleneck(in_channels=512, out_channels=512, start_dw_kernel_size=5, middle_dw_kernel_size=0, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
        UniversalInvertedBottleneck(in_channels=512, out_channels=512, start_dw_kernel_size=5, middle_dw_kernel_size=0, middle_dw_downsample=True, stride=1, expand_ratio=4.0),
    )

    self._block5 = nn.Sequential(
        convbn(in_channels=512, out_channels=960, kernel_size=1, stride=1),
        convbn(in_channels=960, out_channels=1280, kernel_size=1, stride=1),
        nn.AdaptiveAvgPool2d(1)
    )

  def forward(self, x:Tensor) -> List[Tensor]:
    x0 = self._block0(x)
    x1 = self._block1(x0)
    x2 = self._block2(x1)
    x3 = self._block3(x2)
    x4 = self._block4(x3)
    x5 = self._block5(x4)
    return [x1, x2, x3, x4, x5]


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
