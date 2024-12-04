from tinygrad import nn, Tensor
from typing import Optional
from util.shape import make_divisible

class UniversalInvertedBottleneck:
  def __init__(self,
               in_channels:int,
               out_channels:int,
               start_dw_kernel_size:Optional[int]=None, 
               middle_dw_kernel_size:Optional[int]=None, 
               middle_dw_downsample:bool=False,
               stride:int=1,
               expand_ratio:float=1.0
        ):
    self._start_dw = None
    self._middle_dw = None
    if start_dw_kernel_size:
      stride_ = stride if not middle_dw_downsample else 1
      self._start_dw = [
          nn.Conv2d(in_channels, in_channels, kernel_size=start_dw_kernel_size, stride=stride_, groups=in_channels, bias=False),
          nn.BatchNorm2d(in_channels)
      ]
    expand_channels = make_divisible(in_channels * expand_ratio, 8)
    self._expand = [
        nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(expand_channels)
    ]
    if middle_dw_kernel_size:
      stride_ = stride if middle_dw_downsample else 1
      self._middle_dw = [
          nn.Conv2d(expand_channels, expand_channels, kernel_size=middle_dw_kernel_size, stride=stride_, groups=expand_channels, bias=False),
          nn.BatchNorm2d(expand_channels)
      ]
    self._pw = [
        nn.Conv2d(expand_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels)
    ]

  def __call__(self, x:Tensor) -> Tensor:
    if self._start_dw:
      x = x.sequential(self._start_dw)
    x = x.sequential(self._expand).relu6()
    if self._middle_dw:
      x = x.sequential(self._middle_dw).relu6()
    x = x.sequential(self._pw)
    return x
