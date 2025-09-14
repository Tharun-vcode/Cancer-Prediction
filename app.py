import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)  # Match the trained ResNet18 model
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load("D:\\a\\skin_cancer_model_improved.pth", map_location=device))
model.to(device)
model.eval()

# Transforms (match training size)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class names (sorted order to match training)
class_names = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
               'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']

def predict(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
    return pred, prob[0][pred].item()

def generate_gradcam(image, model, target_layer, pred_class):
    cam = GradCAM(model=model, target_layers=[target_layer])
    img_tensor = transform(image).unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    img_array = np.array(image.resize((128, 128))) / 255.0
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    return visualization

# Streamlit app
st.title('Skin Cancer Classifier with Grad-CAM')

uploaded_file = st.file_uploader('Upload a skin lesion image', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict
    pred, prob = predict(image)
    st.write(f'Prediction: {class_names[pred]} (Confidence: {prob:.2f})')

    # Grad-CAM
    target_layer = model.layer4[-1]  # Last conv layer for ResNet18
    cam_image = generate_gradcam(image, model, target_layer, pred)
    st.image(cam_image, caption='Grad-CAM Heatmap Overlay (red areas indicate influencing regions)', use_column_width=True)