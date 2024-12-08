from tinygrad import Tensor

class MNV4LayerScale:
  def __init__(self, init_value:float, embedding_dim:int):
    self._init_value = init_value
    self._embedding_dim = embedding_dim
    self._gamma = Tensor(self._init_value * Tensor.ones(self._embedding_dim,), requires_grad=True)

  def __call__(self, x:Tensor) -> Tensor:
    return x * self._gamma.cast(x.dtype)

