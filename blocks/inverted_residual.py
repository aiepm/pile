from tinygrad import Tensor, nn
from util import make_divisible

class InvertedResidual:
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int,  stride:int=1, expand_ratio:float=1, activation:bool=False, squeeze_excite:bool=False, se_ratio:float=1):
    self._expand = None
    expanded_channels = in_channels
    
    assert stride in [1, 2]
    if stride == 2:
      pad = [(kernel_size-1)//2-1, (kernel_size-1)//2]*2
    else:
      pad = [(kernel_size-1)//2]*4

    if expand_ratio != 1:
      expanded_channels = make_divisible(expand_ratio * in_channels, 8)
      self._expand = [
          nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
          nn.BatchNorm2d(expanded_channels)
      ]

    self._dw = [
        nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, padding=pad, stride=stride, groups=expanded_channels, bias=False),
        nn.BatchNorm2d(expanded_channels)
    ]

    self._has_se = squeeze_excite
    if self._has_se:
      s_chan = make_divisible(expanded_channels * se_ratio, 8)
      self._squeeze = nn.Conv2d(expanded_channels, s_chan, kernel_size=1, bias=True)
      self._excite = nn.Conv2d(s_chan, expanded_channels, kernel_size=1, bias=True)

    self._pw = [
        nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels)
    ]

    self._final_act = activation
    self._use_res_conn = stride = 1 and in_channels == out_channels

  def __call__(self, x:Tensor) -> Tensor:
    ox = x
    if self._expand:
      x = x.sequential(self._expand).relu6()
    x = x.sequential(self._dw).relu6()
    if self._has_se:
      xse = x.avg_pool2d(x.shape[2:4])
      xse = self._squeeze(xse).relu6()
      xse = self._excite(xse)
      x = x.mul(xse.sigmoid())
    x = x.sequential(self._pw)
    if self._final_act:
      x = x.relu6()
    if self._use_res_conn:
      x = x + ox
    return x

