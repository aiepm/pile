from tinygrad import Tensor

class StochasticDepth:
  def __init__(self, survival_prob: float):
    self.survival_prob = survival_prob

  def __call__(self, x:Tensor) -> Tensor:
    if not Tensor.training or self.survival_prob == 1.0:
      return x

    # Generate a random tensor for dropping
    survival_mask = Tensor.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
    x = x.div(self.survival_prob)
    return x * survival_mask

