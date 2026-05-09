import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# -----------------------------
# GPU SETUP
# -----------------------------
print("Checking GPU availability...\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("GPU is AVAILABLE")
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("GPU NOT available, using CPU")

# -----------------------------
# CONFIG
# -----------------------------
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.00001

BASE_PATH = r"C:\Users\Libin Josuva\Downloads\data\archive (3)\Nutrition_dataset"

train_dir = os.path.join(BASE_PATH, "train")
valid_dir = os.path.join(BASE_PATH, "valid")
test_dir  = os.path.join(BASE_PATH, "test")

# -----------------------------
# TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# -----------------------------
# DATASET
# -----------------------------
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# DATASET DISTRIBUTION GRAPH
# -----------------------------
class_counts = {}
for _, label in train_dataset.samples:
    class_name = train_dataset.classes[label]
    class_counts[class_name] = class_counts.get(class_name, 0) + 1

plt.figure()
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=45)
plt.title("Dataset Class Distribution")
plt.show()

# -----------------------------
# MODEL (ResNet50)
# -----------------------------
model = models.resnet50(weights="IMAGENET1K_V1")

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# -----------------------------
# TRAINING
# -----------------------------
train_acc_list = []
valid_acc_list = []
train_loss_list = []
valid_loss_list = []

for epoch in range(EPOCHS):

    # TRAIN
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total
    train_loss = train_loss / len(train_loader)

    # VALIDATION
    model.eval()
    valid_correct = 0
    valid_total = 0
    valid_loss = 0

    with torch.no_grad():
        for images, labels in valid_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()

    valid_acc = 100 * valid_correct / valid_total
    valid_loss = valid_loss / len(valid_loader)

    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)

    scheduler.step()

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("Train Accuracy:", train_acc)
    print("Validation Accuracy:", valid_acc)
    print("Train Loss:", train_loss)
    print("Validation Loss:", valid_loss)

# -----------------------------
# ACCURACY GRAPH
# -----------------------------
plt.figure()
plt.plot(train_acc_list, label="Train Accuracy")
plt.plot(valid_acc_list, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

# -----------------------------
# LOSS GRAPH
# -----------------------------
plt.figure()
plt.plot(train_loss_list, label="Train Loss")
plt.plot(valid_loss_list, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# -----------------------------
# TESTING
# -----------------------------
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# -----------------------------
# METRICS
# -----------------------------
accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100

precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

num_classes = len(train_dataset.classes)

mse = np.mean((np.array(all_preds) - np.array(all_labels))**2) / (num_classes**2)
rmse = math.sqrt(mse)
mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))

print("\n--- TEST METRICS ---")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "cnn_model.pth")

print("Model Saved Successfully!")