
import albumentations as A
import timm
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter

# µ¥ÀÌÅÍ Áõ°­ ¼³Á¤
augmentation_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.6),
    A.Resize(224, 224),
    A.Normalize()
])

# Å¬·¡½ºº° ÀÌ¹ÌÁö ¼ö °è»ê
class_image_counts = Counter(train_labels)
max_class_count = max(class_image_counts.values())

# Å¬·¡½ºº° Áõ°­ È½¼ö ¼³Á¤
augmentation_counts = {cls: max_class_count // count for cls, count in class_image_counts.items()}


# µ¥ÀÌÅÍ¼Â Å¬·¡½º
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, augmentation_counts=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.augmentation_counts = augmentation_counts

    def __len__(self):
        if self.augmentation_counts:
            return sum([self.augmentation_counts[label] for label in self.labels])
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx % len(self.images)]
        label = self.labels[idx % len(self.labels)]
        if self.transform and self.augmentation_counts:
            for _ in range(self.augmentation_counts[label]):
                image = self.transform(image=image)["image"]
        return image, label


# train_images, train_labels¸¦ ÇÐ½À µ¥ÀÌÅÍ¿Í °ËÁõ µ¥ÀÌÅÍ·Î ºÐÇÒ
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2,
                                                                      random_state=42, stratify=train_labels)

train_data = CustomDataset(train_images, train_labels, transform=augmentation_transforms,
                           augmentation_counts=augmentation_counts)
val_data = CustomDataset(val_images, val_labels, transform=augmentation_transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# ¸ðµ¨ ¹× ¼Õ½Ç ÇÔ¼ö ¼³Á¤
model = timm.create_model('resnet18', num_classes=19, pretrained=True).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ÇÐ½À ¹× °ËÁõ °úÁ¤
num_epochs = 30
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == targets).sum().item()

    train_accuracy = train_correct / len(train_data)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == targets).sum().item()

    val_accuracy = val_correct / len(val_data)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, '
          f'Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy * 100:.2f}%')
        # Forward pass
