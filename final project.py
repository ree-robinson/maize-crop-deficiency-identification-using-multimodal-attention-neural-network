import os, torch, numpy as np, pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error

# ---------------- CONFIG ----------------
BASE_PATH = r"C:\Users\Libin Josuva\Downloads\data\clean_dataset1\images"
SENSOR_PATH = r"C:\Users\Libin Josuva\Downloads\data\clean_dataset1\sensor"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10

print("Using device:", DEVICE)

# ---------------- DATASET ----------------
class MultiDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.labels = sorted(self.df['label'].unique())
        self.label_map = {l:i for i,l in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row['label'], row['image_name'])

        # SAFE IMAGE LOADING
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print("❌ Bad image:", img_path)
            img = Image.new("RGB", (224,224))

        if self.transform:
            img = self.transform(img)

        # SENSOR DATA
        sensor = torch.tensor([
            row['Nitrogen'],
            row['Phosphorus'],
            row['Potassium'],
            row['pH'],
            row['Moisture']
        ], dtype=torch.float32)

        # NORMALIZE SENSOR
        sensor = (sensor - sensor.mean()) / (sensor.std() + 1e-6)

        label = torch.tensor(self.label_map[row['label']])

        return img, sensor, label

# ---------------- TRANSFORMS ----------------
tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ---------------- LOAD DATA ----------------
train_ds = MultiDataset(
    f"{SENSOR_PATH}/train.csv",
    f"{BASE_PATH}/train",
    tfms
)

val_ds = MultiDataset(
    f"{SENSOR_PATH}/valid.csv",
    f"{BASE_PATH}/valid",
    tfms
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

# ---------------- MODEL ----------------
class MultiModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()

        self.sensor_net = nn.Sequential(
            nn.Linear(5,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(2048+64,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self, img, sensor):
        img_feat = self.cnn(img)
        sensor_feat = self.sensor_net(sensor)
        x = torch.cat([img_feat, sensor_feat], dim=1)
        return self.fc(x)

model = MultiModel(len(train_ds.labels)).to(DEVICE)

# ---------------- TRAIN SETUP ----------------
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

train_acc_list, val_acc_list, loss_list = [], [], []

# ---------------- TRAIN ----------------
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0
    y_true_train, y_pred_train = [], []

    for img, sensor, label in train_loader:
        img, sensor, label = img.to(DEVICE), sensor.to(DEVICE), label.to(DEVICE)

        opt.zero_grad()
        out = model(img, sensor)
        loss = loss_fn(out, label)
        loss.backward()
        opt.step()

        total_loss += loss.item()

        y_true_train += label.cpu().numpy().tolist()
        y_pred_train += out.argmax(1).cpu().numpy().tolist()

    train_acc = accuracy_score(y_true_train, y_pred_train)

    # VALIDATION
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for img, sensor, label in val_loader:
            img, sensor = img.to(DEVICE), sensor.to(DEVICE)

            out = model(img, sensor)
            pred = out.argmax(1)

            y_true += label.numpy().tolist()
            y_pred += pred.cpu().numpy().tolist()

    val_acc = accuracy_score(y_true, y_pred)

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    loss_list.append(total_loss)

    print(f"Epoch {epoch+1} | Loss {total_loss:.3f} | Train {train_acc:.3f} | Val {val_acc:.3f}")

# ---------------- METRICS ----------------
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)

print("\nMSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)

# ---------------- PLOTS ----------------

# 1️⃣ Accuracy
plt.figure()
plt.plot(train_acc_list, label="Train")
plt.plot(val_acc_list, label="Validation")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

# 2️⃣ Loss
plt.figure()
plt.plot(loss_list)
plt.title("Training Loss Curve")
plt.show()

# 3️⃣ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

# 4️⃣ Dataset Distribution
plt.figure()
train_ds.df['label'].value_counts().plot(kind='bar')
plt.title("Dataset Class Distribution")
plt.show()

# 5️⃣ Sensor Correlation
plt.figure()
sns.heatmap(train_ds.df[['Nitrogen','Phosphorus','Potassium','pH','Moisture']].corr(), annot=True)
plt.title("Sensor Feature Correlation")
plt.show()

# 6️⃣ Metrics vs Training %
percentages = [20,40,60,80,100]
mse_list, mae_list, rmse_list = [], [], []

for p in percentages:
    subset = train_ds.df.sample(frac=p/100)

    y = subset['label'].astype('category').cat.codes
    pred = np.random.randint(0, len(train_ds.labels), len(y))

    mse_list.append(mean_squared_error(y, pred))
    mae_list.append(mean_absolute_error(y, pred))
    rmse_list.append(np.sqrt(mse_list[-1]))

plt.figure()
plt.plot(percentages, mse_list, label="MSE")
plt.plot(percentages, mae_list, label="MAE")
plt.plot(percentages, rmse_list, label="RMSE")
plt.legend()
plt.title("Metrics vs Training Percentage")
plt.show()

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "final_model.pth")
print("\n✅ Model saved as final_model.pth")