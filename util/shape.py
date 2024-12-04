from typing import Optional

def make_divisible(x:float, d:int, minval:Optional[int]=None) -> int:
  if minval is None:
    minval = d
  return max(minval, int(x + d // 2) // d)


