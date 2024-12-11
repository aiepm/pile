import os
from PIL import Image

class CIFAR10:
  def __init__(self, input_dir:str):
    self._image_dir = os.path.join(input_dir, 'images')
    self._labels = []
    with open(os.path.join(input_dir, 'labels.txt'), 'r') as f:
      self._labels = [line.split() for line in f.readlines()]

  def __len__(self):
    return len(self._labels)

  def __getitem__(self, idx):
    filename, label = self._labels[idx]
    return Image.open(os.path.join(self._image_dir, filename)), int(label)
