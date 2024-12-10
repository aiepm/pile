from tinygrad import Tensor, nn
from blocks.mhsa import MHSA
from blocks.universal_inverted_bottleneck import UniversalInvertedBottleneck
from blocks.inverted_residual import InvertedResidual
from specs import MODEL_SPECS

def build_blocks(layer_spec):
  if not layer_spec.get('block_name'):
    return []
  block_names = layer_spec['block_name']
  layers = []
  if block_names == "convbn":
    schema_ = ['in_channels', 'out_channels', 'kernel_size', 'stride']
    for i in range(layer_spec['num_blocks']):
      args = dict(zip(schema_, layer_spec['block_specs'][i]))
      block = [
        nn.Conv2d(**args),
        nn.BatchNorm2d(args['out_channels']),
        lambda x: x.relu6()
      ]
      layers.extend(block)
  elif block_names == "uib":
    schema_ =  ['in_channels', 'out_channels', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride', 'expand_ratio', 'mhsa']
    for i in range(layer_spec['num_blocks']):
      args = dict(zip(schema_, layer_spec['block_specs'][i]))
      mhsa = args.pop("mhsa") if "mhsa" in args else 0
      layers.append(UniversalInvertedBottleneck(**args))
      if mhsa:
        mhsa_schema_ = [
            "input_dim", "num_heads", "key_dim", "value_dim", "query_h_strides", "query_w_strides", "kv_strides", 
            "use_layer_scale", "use_multi_query", "use_residual"
        ]
        args = dict(zip(mhsa_schema_, [args['out_channels']] + (mhsa)))
        layers.append(MHSA(**args))
  elif block_names == "fused_ib":
    schema_ = ['in_channels', 'out_channels', 'stride', 'expand_ratio', 'activation']
    for i in range(layer_spec['num_blocks']):
      args = dict(zip(schema_, layer_spec['block_specs'][i]))
      layers.append(InvertedResidual(**args))
  else:
    raise NotImplementedError
  return layers

class MobilenetV4:
  def __init__(self, sspec):
    self._sspec = sspec
    self._spec = MODEL_SPECS[self.model]
    
    self._conv0 = build_blocks(self._spec['conv0'])
    self._layer1 = build_blocks(self._spec['layer1'])
    self._layer2 = build_blocks(self._spec['layer2'])
    self._layer3 = build_blocks(self._spec['layer3'])
    self._layer4 = build_blocks(self._spec['layer4'])
    self._layer5 = build_blocks(self._spec['layer5'])

  def __call__(self, x:Tensor) -> Tensor:
    x0 = x.sequential(self._conv0)
    x1 = x0.sequential(self._layer1)
    x2 = x1.sequential(self._layer2)
    x3 = x2.sequential(self._layer3)
    x4 = x3.sequential(self._layer4)
    x5 = x4.sequential(self._layer5)
    return [x1, x2, x3, x4, x5]
