import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import os

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 CHANGE THIS PATH IF NEEDED
MODEL_PATH = r"C:\Users\Libin Josuva\PycharmProjects\PythonProject\.venv\final_model.pth"

# ---------------- MODEL ----------------
class MultiModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = models.resnet50(weights=None)
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

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at:\n{MODEL_PATH}")
        st.stop()

    model = MultiModel(num_classes=5)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model

model = load_model()

# ---------------- TRANSFORM ----------------
tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ---------------- LABELS ----------------
labels = [
    "all absent",
    "all present",
    "nitrogen deficiency",
    "phosphorus deficiency",
    "potassium deficiency"
]

# ---------------- UI ----------------
st.set_page_config(page_title="Maize AI", layout="centered")

st.title("🌽 Maize Nutrient Detection")
st.write("Upload image + enter sensor values")

# IMAGE INPUT
img_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

# SENSOR INPUT
N = st.slider("Nitrogen", 0, 150, 80)
P = st.slider("Phosphorus", 0, 100, 50)
K = st.slider("Potassium", 0, 100, 50)
pH = st.slider("pH", 4.0, 9.0, 6.5)
M = st.slider("Moisture", 0, 100, 60)

# ---------------- PREDICT ----------------
if st.button("Predict"):

    if img_file is None:
        st.warning("⚠️ Please upload an image")
    else:
        try:
            img = Image.open(img_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)

            img_tensor = tfms(img).unsqueeze(0).to(DEVICE)

            sensor = torch.tensor([N,P,K,pH,M], dtype=torch.float32)
            sensor = (sensor - sensor.mean()) / (sensor.std() + 1e-6)
            sensor = sensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(img_tensor, sensor)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]

            st.subheader("Prediction Confidence")

            for i, p in enumerate(probs):
                st.write(f"{labels[i]} : {p:.4f}")

            st.success(f"✅ Final Prediction: {labels[np.argmax(probs)]}")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")