from torch import nn, Tensor
from util import make_divisible

class InvertedResidual(nn.Module):
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int,  stride:int=1, expand_ratio:float=1, activation:bool=False, squeeze_excite:bool=False, se_ratio:float=1):
    super().__init__()
    expanded_channels = in_channels
    
    assert stride in [1, 2]
    pad = [(kernel_size-1)//2]*4 if stride == 1 else [(kernel_size-1)//2-1, (kernel_size-1)//2]*2
    
    self._expand = nn.Sequential()
    if expand_ratio != 1:
      expanded_channels = make_divisible(expand_ratio * in_channels, 8)
      self._expand = nn.Sequential(
          nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
          nn.BatchNorm2d(expanded_channels),
          nn.ReLU6()
      )

    self._dw = nn.Sequential(
        nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, padding=pad, stride=stride, groups=expanded_channels, bias=False),
        nn.BatchNorm2d(expanded_channels),
        nn.ReLU6()
    )

    self._se = nn.Sequential()
    if squeeze_excite:
      s_chan = make_divisible(expanded_channels * se_ratio, 8)
      self._se = lambda x: x.mul(nn.Sequential(
          nn.AvgPool2d(x.shape[2:4]),
          nn.Conv2d(expanded_channels, s_chan, kernel_size=1, bias=True),
          nn.ReLU6(),
          nn.Conv2d(s_chan, expanded_channels, kernel_size=1, bias=True),
          nn.Sigmoid()
      )(x))

    self._pw = nn.Sequential(
        nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6() if activation else nn.Identity()
    )

    self._use_res_conn = stride = 1 and in_channels == out_channels

  def forward(self, x:Tensor) -> Tensor:
    ox = x
    x = self._expand(x)
    x = self._dw(x)
    x = self._se(x)
    x = self._pw(x)
    x = (x + ox) if self._use_res_conn else x
    return x

