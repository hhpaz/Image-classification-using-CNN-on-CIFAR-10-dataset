import torch
from torch.utils.data import DataLoader ,Dataset
import torch.nn as nn
from torchvision import transforms
import torchvision


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
      ])

train_set = torchvision.datasets.CIFAR10(root= './data' ,transform = transform  , train=True , download=True )
test_set = torchvision.datasets.CIFAR10(root= './data' , transform = transform  , train=False ,download=True  )

train_loader = DataLoader(train_set , batch_size=64 , shuffle=True)
test_loader = DataLoader(test_set , batch_size=64 , shuffle=True)

# حلقه آموزش

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # فوروارد
        outputs = model(images)
        loss = criterion(outputs, labels)

        # بک‌پراپاگیشن
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_set)
    epoch_acc = 100.0 * correct / total

    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# حلقه تست
model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss = test_loss / len(test_set)
test_acc = 100.0 * correct / total

print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')