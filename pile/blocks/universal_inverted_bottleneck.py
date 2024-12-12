from torch import nn, Tensor
from typing import Optional
from pile.util.shape import make_divisible

class UniversalInvertedBottleneck(nn.Module):
  def __init__(self,
               in_channels:int,
               out_channels:int,
               start_dw_kernel_size:Optional[int]=None, 
               middle_dw_kernel_size:Optional[int]=None, 
               middle_dw_downsample:bool=False,
               stride:int=1,
               expand_ratio:float=1.0
        ):
    super().__init__()
    self._start_dw = nn.Sequential()
    if start_dw_kernel_size:
      stride_ = stride if not middle_dw_downsample else 1
      self._start_dw = nn.Sequential(
          nn.Conv2d(in_channels, in_channels, kernel_size=start_dw_kernel_size, stride=stride_, groups=in_channels, bias=False),
          nn.BatchNorm2d(in_channels)
      )
    
    expand_channels = make_divisible(in_channels * expand_ratio, 8)
    self._expand = nn.Sequential(
        nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(expand_channels),
        nn.ReLU6()
    )

    self._middle_dw = nn.Sequential()
    if middle_dw_kernel_size:
      stride_ = stride if middle_dw_downsample else 1
      self._middle_dw = nn.Sequential(
          nn.Conv2d(expand_channels, expand_channels, kernel_size=middle_dw_kernel_size, stride=stride_, groups=expand_channels, bias=False),
          nn.BatchNorm2d(expand_channels),
          nn.ReLU6()
      )

    self._pw = nn.Sequential(
        nn.Conv2d(expand_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels)
    )

  def __call__(self, x:Tensor) -> Tensor:
    x = self._start_dw(x)
    x = self._expand(x)
    x = self._middle_dw(x)
    x = self._pw(x)
    return x
