import torch

# Contains all the specific functions related to neural networks
import torch.nn as nn

#This module provides optimization algorithms to update model weights during training
import torch.optim as optim

# Dataset lets you define how to load your data, and DataLoader efficiently feeds that data to your model in batches during training.
from torch.utils.data import Dataset,DataLoader

# provides image preprocessing functions (like resizing, normalizing, and converting to tensors) for training deep learning models.
import torchvision.transforms as transforms

# ImageFolder loads images from a directory structure where subfolders are class labels, making it easy to build image datasets for training models.
from torchvision.datasets import ImageFolder

# is a PyTorch library offering a huge collection of pretrained image models (like CNNs and Vision Transformers) with a clean and efficient API for training and fine-tuning.
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import streamlit as st

import kagglehub
from tqdm import tqdm

# Download latest version
path = kagglehub.dataset_download("gpiosenka/cards-image-datasetclassification")

print("Path to dataset files:", path)

class PlayingCardDataset(Dataset):
    def __init__(self,data_dir,transform = None):
        self.data = ImageFolder(data_dir,transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

dataset = PlayingCardDataset(data_dir=r"C:\Users\Prathmesh\.cache\kagglehub\datasets\gpiosenka\cards-image-datasetclassification\versions\2")

# Proper path string
data_dir = r'C:\Users\Prathmesh\.cache\kagglehub\datasets\gpiosenka\cards-image-datasetclassification\versions\2\train'

# Create ImageFolder instance
folder = ImageFolder(data_dir)

# Reverse class_to_idx mapping
target_to_class = {v: k for k, v in folder.class_to_idx.items()}
print(target_to_class)


# Transform Pipeline : uses Torchvision.transforms

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = PlayingCardDataset(data_dir,train_transform)

# Batching

dataloader = DataLoader(dataset,batch_size=32,shuffle=True)

for images, labels in dataloader:
    break

print(images.shape)

# Creating a pytorch model

class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes = 53):
        super(SimpleCardClassifier,self).__init__()
        self.base_model = timm.create_model('efficientnet_b0',pretrained = True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280

        # Classifier
        self.classifier = nn.Linear(enet_out_size,num_classes)

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.classifier(x)

model = SimpleCardClassifier(num_classes=53)
print(model(images))

# Training loop

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(),lr = 0.001)

val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_folder = r'C:\Users\Prathmesh\.cache\kagglehub\datasets\gpiosenka\cards-image-datasetclassification\versions\2\train'
valid_folder = r'C:\Users\Prathmesh\.cache\kagglehub\datasets\gpiosenka\cards-image-datasetclassification\versions\2\valid'
test_folder = r'C:\Users\Prathmesh\.cache\kagglehub\datasets\gpiosenka\cards-image-datasetclassification\versions\2\test'

train_dataset = PlayingCardDataset(train_folder, transform=train_transform)
val_dataset = PlayingCardDataset(valid_folder, transform=val_test_transform)
test_dataset = PlayingCardDataset(test_folder, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Simple training loop
num_epochs = 20
train_losses, val_losses = [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SimpleCardClassifier(num_classes=53)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if os.path.exists("card_classifier.pth"):
    model.load_state_dict(torch.load("card_classifier.pth", map_location=device))
    model.eval()
    print("Model loaded from file.")
else:
    print("Training model...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc='Training loop'):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation loop'):
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

    torch.save(model.state_dict(), "card_classifier.pth")
    print("Model saved.")

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.title("Loss over epochs")
plt.show()

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# Load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)


# Predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


# Visualization
def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    # Display image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")

    # Display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()


# Example usage
test_image = r"C:\Users\Prathmesh\Downloads\testcard.png"
transform = val_test_transform

original_image, image_tensor = preprocess_image(test_image, transform)
probabilities = predict(model, image_tensor, device)


class_names = train_dataset.classes
visualize_predictions(original_image, probabilities, class_names)