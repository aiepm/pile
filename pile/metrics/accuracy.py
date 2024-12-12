import torch
from torch import nn

def calculate_accuracy(loader:torch.utils.data.DataLoader, model:nn.Module, device:str='cuda'):
  total = 0
  correct = 0
  for images, labels in loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  return 100 * correct / total

