import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import timm
import os
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Loading model
@st.cache_resource
def load_model():
    model = SimpleCardClassifier()
    model.load_state_dict(torch.load("card_classifier.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# Load class names
@st.cache_data
def get_class_names():
    dummy_path = r'C:\Users\Prathmesh\.cache\kagglehub\datasets\gpiosenka\cards-image-datasetclassification\versions\2\train'
    folder = ImageFolder(dummy_path)
    return folder.classes

# Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# Prediction
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
    return probs.cpu().numpy().flatten()

model = load_model()
class_names = get_class_names()

# Ui hai
st.title("🃏 Playing Card Classifier")
st.markdown("---")
uploaded = None
option = st.radio("Choose input method: ",["Upload Image","Use Camera","Live Prediction (CV)"])

if option == "Upload Image":

    uploaded = st.file_uploader("Upload a card image", type=["jpg", "png", "jpeg"])

elif option == "Use Camera":
    uploaded = st.camera_input("Take a picture")

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")
    st.image(image,caption="input image",width = 250)

    image_tensor = transform(image).unsqueeze(0).to(device)

    probs = predict(model,image_tensor)

    # Top prediction
    top_idx = probs.argmax()
    class_name = class_names[top_idx]
    confidence = probs[top_idx]

    st.success(f"🃏 Predicted: **{class_name}** ({confidence * 100:.2f}%)")

    # top-3 predictionsss
    topk = torch.topk(torch.tensor(probs), 3)
    st.subheader("Top-3 Predictions")
    for i in range(3):
        idx = topk.indices[i].item()
        st.write(f"{class_names[idx]}: {topk.values[i].item() * 100:.2f}%")

elif option == "Live Prediction (CV)":
    import cv2
    import time

    stframe = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        run = st.checkbox("Start Live Prediction")
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)


            if len(faces) > 0:
                cv2.putText(frame, "🃏 Joker Detected!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame, channels="RGB")
                time.sleep(0.1)
                continue

            # Preprocess
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img).convert("RGB")
            image_tensor = transform(pil_img).unsqueeze(0).to(device)

            # Predict
            probs = predict(model, image_tensor)
            top_idx = probs.argmax()
            class_name = class_names[top_idx]
            confidence = probs[top_idx]


            cv2.putText(frame, f"{class_name} ({confidence * 100:.2f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")

            time.sleep(0.05)

        cap.release()

#About the App
with st.expander("📄 About This App"):
    st.markdown("""
    This is an intelligent Playing Card Classifier built with:
    - **PyTorch** (EfficientNet-B0)
    - **Streamlit**
    - **Computer Vision**

    Dataset used: [Kaggle Playing Cards Dataset](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)

    Model was trained for multi-class classification with 53 classes.
    """)

# --- Sidebar Info (Optional) ---
# st.sidebar.header("📊 Model Info")
# st.sidebar.markdown(f"""
# - **Model**: EfficientNet-B0
# - **Classes**: {len(class_names)}
# - **Device**: {"GPU" if torch.cuda.is_available() else "CPU"}
# """)

# --- Footer / Credits ---
st.markdown("---")
st.markdown("""
    <div style='text-align: center; font-size: 15px'>
        Made with ❤️ by <b>Prathmesh Bajpai</b><br>
        <a href='https://github.com/LEADisDEAD' target='_blank'>GitHub</a> | 
        <a href='https://www.linkedin.com/in/prathmesh-bajpai-8429652aa/' target='_blank'>LinkedIn</a>
    </div>
""", unsafe_allow_html=True)
