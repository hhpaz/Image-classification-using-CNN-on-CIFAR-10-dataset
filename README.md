# CNN Image Classifier on CIFAR-10

This project implements a simple Convolutional Neural Network (CNN) in PyTorch to classify images from the CIFAR-10 dataset.

## 🔧 Architecture

- 3 Conv blocks (Conv → ReLU → MaxPool)
- AdaptiveAvgPool2D to (7×7)
- Fully Connected (128×7×7 → 512 → 10)

## 📦 Dataset

- CIFAR-10
- 60,000 images (32×32, 10 classes)

## 🚀 How to Run

```bash
pip install -r requirements.txt
python train.py
# Image-classification-using-CNN-on-CIFAR-10-dataset
