# Card Sense - Real-Time Playing Card Classifier 🃏

Card Sense is an image classification system built using PyTorch, EfficientNet, and Streamlit, designed to accurately identify standard playing cards from images. It supports multiple input modes including image upload, camera capture, and live real-time prediction via OpenCV.

### ⚙️ Hardware Acceleration
This project was developed and trained on a system with an NVIDIA RTX 3060 GPU, utilizing:

- CUDA Toolkit for GPU acceleration

- cuDNN for optimized deep learning performance

- End-to-end training and inference performed on GPU (not CPU), ensuring faster convergence and efficient real-time predictions


---

## 🔧 Features

- 📸 Upload or capture card images
- 🎥 Live webcam-based card prediction
- 🔍 Displays top-3 predicted cards with confidence
- 🧠 EfficientNet-B0 deep learning model
- 🌐 Streamlit-based interactive UI
- ⚡ GPU-accelerated training and inference (CUDA supported)

---

## 🧠 How It Works

### 1. Data Preparation
- Uses ImageFolder format: images are stored in subfolders named after class labels.
- Applied transforms: resize to 128x128, normalize with ImageNet stats, augment (rotation, jitter).
- Dataset split into train/validation/test directories for better model evaluation.
  
### 2. Model Architecture
- Based on EfficientNet-B0 from the timm library – pretrained and efficient.
- Final classification layer changed to Linear(1280, 53) for 53 card classes.
- Feature extraction via nn.Sequential, followed by flattening and classification.

### 3.Model Training (main.py)
- Trains for multiple epochs with CrossEntropyLoss and Adam optimizer.
- Each epoch runs training and validation loops with real-time progress (via tqdm).
- Saves the trained model as card_classifier.pth for future use.

### 4.Prediction Pipeline (app.py)
- Loads trained model and class names at startup using @st.cache_resource.
- Supports image upload, camera input, or live webcam stream for predictions.
- Preprocesses image, feeds into model, and outputs top predictions with confidence.

### 5.Streamlit Web App
- Clean UI with radio buttons to choose input mode.
- Displays image preview, predicted class, and top-3 confidence scores.
- For live mode, uses OpenCV window to show predictions in real-time.

---

## 🗂 Folder Structure

```
Card-Sense/
│
├── main.py                 # Training script
├── app.py                  # Streamlit web app
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and instructions
├── .gitignore              # Ignored files/folders
└── card_classifier.pth     # (Optional) Saved trained model file

```
---
## Getting Started

### 1. Clone the Repository
```
git clone https://github.com/LEADisDEAD/Card-Sense.git
cd Card-Sense
```
### 2. Create & Activate a Virtual Environment (Optional)
```
python -m venv .venv
source .venv/bin/activate  # For Unix/macOS
.venv\Scripts\activate     # For Windows
```
### 3. Install Dependencies
```
pip install -r requirements.txt
```
### 4. Run the Web App
```
streamlit run app.py
```
---
## 🔧 Prerequisites

To train the model from scratch:

- Python 3.8+
- PyTorch with CUDA enabled
- CUDA Toolkit installed (Installation Guide)
- cuDNN installed and properly configured

---
### 📸 Dataset

Dataset used: Kaggle - Playing Cards Dataset
You can automatically download it via kagglehub in the main.py script.
Ensure your Kaggle API token is configured.

---

## 🙋‍♂️ Author - Prathmesh Bajpai

- LinkedIn: [linkedin.com/in/prathmesh-bajpai-8429652aa](https://www.linkedin.com/in/prathmesh-bajpai-8429652aa)




