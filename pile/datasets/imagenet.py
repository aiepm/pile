import cv2
import os
from torch.utils.data import Dataset


class ImageNet1KDataset(Dataset):
  def __init__(self, root_dir, transform=None):
    """
    Args:
        root_dir (str): Root directory containing 'train' or 'val' folders.
        transform (callable, optional): Transformations to apply to images.
    """
    self.root_dir = root_dir
    self.transform = transform
    self.image_paths = []
    self.labels = []

    # Load image paths and labels
    for class_label in os.listdir(root_dir):
      class_folder = os.path.join(root_dir, class_label)
      if os.path.isdir(class_folder):
        for image_name in os.listdir(class_folder):
          image_path = os.path.join(class_folder, image_name)
          self.image_paths.append(image_path)
          self.labels.append(int(class_label))  # Class labels are folder names

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    label = self.labels[idx]

    # Load image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply transformations
    if self.transform:
      image = self.transform(image=image)['image']

    return image, label

