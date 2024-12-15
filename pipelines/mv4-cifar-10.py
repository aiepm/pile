import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms

from pile.metrics import calculate_accuracy
from pile.models.mobilenet_v4 import MobilenetV4ConvLarge
from pile.schedulers import WarmupCosineScheduler
from torch import optim, Tensor
from torch.utils.data import DataLoader
from pile.util import get_current_lr

BATCH_SIZE = 1024
NUM_EPOCHS = 1000
WARMUP_EPOCHS = 10
DEVICE_NAME = 'cuda:0'

class TModel(nn.Module):
  def __init__(self, backbone, dropout=0.2):
    super().__init__()
    self.backbone = backbone
    self.classifier_head = nn.Sequential(
        nn.Linear(1280, 512),
        nn.BatchNorm1d(512),
        nn.ReLU6(),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU6(),
        nn.Linear(256, 10)
    )

  def forward(self, x:Tensor) -> Tensor:
    x = self.backbone(x)[-1]
    x = x.view(x.shape[0], -1)
    x = self.classifier_head(x)
    return x


def main():
  # Check if GPU is available
  device = torch.device(DEVICE_NAME if torch.cuda.is_available() else 'cpu')
  print(device)
  
  # Data transformations
  transform = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop((32, 32), padding=2),
    transforms.Resize((128, 128), transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])
  
  test_transform = transforms.Compose([
    transforms.Resize((128, 128), transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])

  train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, pin_memory_device=DEVICE_NAME, persistent_workers=True, prefetch_factor=4)
  test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, pin_memory_device=DEVICE_NAME, persistent_workers=True, prefetch_factor=4)

  model = TModel(MobilenetV4ConvLarge(), 0.2)
  model = model.to(device)
  #model = torch.compile(model, mode='max-autotune')

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.04, nesterov=True, momentum=0.9, weight_decay=1e-4)
  scheduler = WarmupCosineScheduler(optimizer, NUM_EPOCHS, WARMUP_EPOCHS)
  #optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
  #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

  train_acc_h = [0.0]
  test_acc_h = [0.0]
  test_loss_h = [0.0]

  scaler = torch.amp.GradScaler(DEVICE_NAME)
  for epoch in range(NUM_EPOCHS):
    model.train()
    current_lr = get_current_lr(optimizer)    
    running_loss = 0.0
    
    for images, labels in train_loader:
      images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
  
      with torch.amp.autocast(DEVICE_NAME):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
      
      # Backward pass and optimization
      optimizer.zero_grad()

      optimizer.zero_grad()
      scaler.scale(loss).backward()

      if epoch > WARMUP_EPOCHS:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      scaler.step(optimizer)
      scaler.update()

      running_loss += loss.item()
  
    avg_train_loss = running_loss / len(train_loader)
    
    if epoch % 4 == 0:
      # Test set evaluation
      model.eval()
      test_loss = 0.0
      with torch.no_grad():
        for images, labels in test_loader:
          images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
          outputs = model(images)
          loss = criterion(outputs, labels)
          test_loss += loss.item()
      
      avg_test_loss = test_loss / len(test_loader)
      test_loss_h.append(avg_test_loss)

      train_accuracy = calculate_accuracy(train_loader, model, device=DEVICE_NAME)
      test_accuracy = calculate_accuracy(test_loader, model, device=DEVICE_NAME)
      train_acc_h.append(train_accuracy)
      test_acc_h.append(test_accuracy)

    # Scheduler step
    #scheduler.step(avg_train_loss)
    scheduler.step()

    train_accuracy = train_acc_h[-1]
    test_accuracy = test_acc_h[-1]
    avg_test_loss = test_loss_h[-1]
        
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], LR: {current_lr:.6f}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.2f}%")

  print('best train accuracy: ', max(train_acc_h))
  print('best test accuracy: ', max(test_acc_h))

main()
