# 🩺 Skin Cancer Classifier with Grad-CAM

This project uses **deep learning (ResNet50 & ResNet18)** to classify 7 types of skin lesions 
from the HAM10000 dataset and visualize decision regions using **Grad-CAM**.

## 🚀 Features
- Fine-tuned **ResNet50 (TensorFlow/Keras)** on HAM10000 (92% validation accuracy).
- Converted to **ResNet18 (PyTorch)** for lightweight deployment.
- Integrated **Grad-CAM visualizations** to interpret predictions.
- **Streamlit app** for interactive image upload & heatmap visualization.

## 🗂 Dataset
[HAM10000 Dataset (Kaggle)](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)

## 📊 Classes
- Actinic keratoses
- Basal cell carcinoma
- Benign keratosis-like lesions
- Dermatofibroma
- Melanoma
- Melanocytic nevi
- Vascular lesions

## ⚡ How to Run
1. Clone the repo
   ```bash
   git clone https://github.com/<your-username>/skin-cancer-classifier.git
   cd skin-cancer-classifier
