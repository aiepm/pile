import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms

from pile.metrics import calculate_accuracy
from pile.models.mobilenet_v4 import MobilenetV4
from pile.schedulers import WarmupCosineScheduler
from torch import optim
from torch.utils.data import DataLoader
from pile.util import get_current_lr

BATCH_SIZE = 512
NUM_EPOCHS = 2000
WARMUP_EPOCHS = 20


def main():
  # Check if GPU is available
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # Data transformations
  transform = transforms.Compose([
      transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.RandomCrop((32, 32), padding=2),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])
  
  test_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])

  train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
  test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

  model = MobilenetV4('MobileNetV4HybridLarge')
  model = model.to(device)
  model = torch.compile(model, mode='reduce-overhead')

  x = torch.rand(1, 3, 224, 224).to(device)

  model(x)

  return

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1, nesterov=True, momentum=0.9, weight_decay=1e-4)
  scheduler = WarmupCosineScheduler(optimizer, NUM_EPOCHS, WARMUP_EPOCHS)
  #optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
  #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)


  # Training loop
  for epoch in range(NUM_EPOCHS):  # Number of epochs
    model.train()  # Set model to training mode
    current_lr = get_current_lr(optimizer)    
    running_loss = 0.0
    
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
  
      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)
      
      # Backward pass and optimization
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  
      running_loss += loss.item()
  
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = calculate_accuracy(train_loader, model)
    
    # Test set evaluation
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
      for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
  
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = calculate_accuracy(test_loader, model)
    
    # Scheduler step
    #scheduler.step(avg_train_loss)
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], LR: {current_lr:.6f}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.2f}%")

main()
