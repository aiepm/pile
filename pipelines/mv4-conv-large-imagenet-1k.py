import os
import torch
import torchvision.transforms.v2 as transforms

from pile.datasets.imagenet import ImageNet1KDataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pile.models.mobilenet_v4 import MobilenetV4ConvLarge
from pile.schedulers import WarmupCosineScheduler
from torch import optim, Tensor
from pile.util import get_current_lr

BATCH_SIZE = 256
NUM_EPOCHS = 5
DEVICE_NAME = 'cuda:0'
METRICS_UPDATE_STEP = 1
NUM_STEPS = 1
WARMUP_STEPS = 1

class TModel(nn.Module):
  def __init__(self, backbone, dropout=0.2):
    super().__init__()
    self.backbone = backbone
    self.classifier_head = nn.Sequential(
      nn.Linear(1280, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU6(),
      nn.Dropout(dropout),
      nn.Linear(1024, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU6(),
      nn.Linear(1024, 1000)
    )

  def forward(self, x: Tensor) -> Tensor:
    x = self.backbone(x)[-1]
    x = x.view(x.shape[0], -1)
    x = self.classifier_head(x)
    return x

def get_imagenet_dataloaders(data_dir, batch_size=32, num_workers=4):
  # Define transforms
  imagenet_mean = [0.485, 0.456, 0.406]
  imagenet_std = [0.229, 0.224, 0.225]

  train_transforms = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
  ])

  val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
  ])

  # Create datasets
  train_dataset = ImageNet1KDataset(
    root_dir=os.path.join(data_dir, 'train'),
    transform=train_transforms
  )
  val_dataset = ImageNet1KDataset(
    root_dir=os.path.join(data_dir, 'val'),
    transform=val_transforms
  )

  # Create dataloaders
  train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
  )
  val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
  )

  return train_loader, val_loader

@torch.no_grad()
def validate(model, dataloader, device, criterion):
  """Evaluate the model on a given dataloader and return avg_loss, top1_acc, top5_acc."""
  model.eval()
  total_loss = 0.0
  total_samples = 0
  top1_correct = 0
  top5_correct = 0

  for images, labels in tqdm(dataloader, 'Validation'):
    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    with torch.amp.autocast(device.type, dtype=torch.float16):
      outputs = model(images)
      loss = criterion(outputs, labels)

    batch_size = labels.size(0)
    total_loss += loss.item() * batch_size
    total_samples += batch_size

    # Calculate top-1 and top-5 accuracies
    _, pred = outputs.topk(5, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    # top-1
    top1_correct += correct[:1].float().sum()
    # top-5
    top5_correct += correct[:5].float().sum()

  avg_loss = total_loss / total_samples
  top1_acc = (top1_correct / total_samples) * 100.0
  top5_acc = (top5_correct / total_samples) * 100.0
  return avg_loss, top1_acc.item(), top5_acc.item()

def main():
  # Check if GPU is available
  device = torch.device(DEVICE_NAME if torch.cuda.is_available() else 'cpu')
  print(device)

  train_loader, test_loader = get_imagenet_dataloaders(
    '/core/datasets/imagenet/target_dir/',
    batch_size=BATCH_SIZE,
    num_workers=8
  )

  global NUM_STEPS, WARMUP_STEPS
  NUM_STEPS = len(train_loader) * NUM_EPOCHS
  WARMUP_STEPS = NUM_STEPS // 100

  model = TModel(MobilenetV4ConvLarge(), 0.2).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(
    model.parameters(),
    lr=0.08,
    nesterov=True,
    momentum=0.9,
    weight_decay=1e-4
  )
  scheduler = WarmupCosineScheduler(optimizer, NUM_STEPS, WARMUP_STEPS)

  train_metrics = {'top1': [], 'top5': [], 'loss': []}
  test_metrics = {'top1': [], 'top5': [], 'loss': []}

  scaler = torch.amp.GradScaler(device.type)
  for epoch in range(NUM_EPOCHS):
    model.train()
    current_lr = get_current_lr(optimizer)
    running_loss = 0.0

    for images, labels in tqdm(train_loader, 'Train'):
      images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

      with torch.amp.autocast(device.type, dtype=torch.float16):
        outputs = model(images)
        loss = criterion(outputs, labels)

      optimizer.zero_grad()
      scaler.scale(loss).backward()
      if epoch > WARMUP_STEPS:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      scaler.step(optimizer)
      scaler.update()

      running_loss += loss.item()

      scheduler.step()

    avg_train_loss = running_loss / len(train_loader)

    # Evaluate on test set every METRICS_UPDATE_STEP
    if epoch % METRICS_UPDATE_STEP == 0:
      avg_test_loss, test_top1, test_top5 = validate(model, test_loader, device, criterion)
      avg_train_loss_eval, train_top1, train_top5 = validate(model, train_loader, device, criterion)

      # Record metrics
      train_metrics['loss'].append(avg_train_loss_eval)
      train_metrics['top1'].append(train_top1)
      train_metrics['top5'].append(train_top5)

      test_metrics['loss'].append(avg_test_loss)
      test_metrics['top1'].append(test_top1)
      test_metrics['top5'].append(test_top5)
    else:
      # If not evaluating this epoch, reuse last metrics for printing
      avg_test_loss = test_metrics['loss'][-1] if len(test_metrics['loss']) > 0 else 0.0
      train_top1 = train_metrics['top1'][-1] if len(train_metrics['top1']) > 0 else 0.0
      train_top5 = train_metrics['top5'][-1] if len(train_metrics['top5']) > 0 else 0.0
      test_top1 = test_metrics['top1'][-1] if len(test_metrics['top1']) > 0 else 0.0
      test_top5 = test_metrics['top5'][-1] if len(test_metrics['top5']) > 0 else 0.0

    print(
      f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
      f"LR: {current_lr:.6f}, "
      f"Train Loss: {avg_train_loss:.4f}, "
      f"Test Loss: {avg_test_loss:.4f}, "
      f"Train Top1: {train_top1:.2f}%, Train Top5: {train_top5:.2f}%, "
      f"Test Top1: {test_top1:.2f}%, Test Top5: {test_top5:.2f}%"
    )

  # Print best results
  best_train_top1 = max(train_metrics['top1']) if train_metrics['top1'] else 0.0
  best_test_top1 = max(test_metrics['top1']) if test_metrics['top1'] else 0.0
  print('Best train top1 accuracy: ', best_train_top1)
  print('Best test top1 accuracy: ', best_test_top1)

if __name__ == "__main__":
  main()

