from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from transformers import ViTForImageClassification
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import os

app = Flask(__name__)

# Load ViT Model
def load_vit_model():
    vit_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )
    vit_model.load_state_dict(torch.load("fake_real_vit1.pth", map_location="cpu"))
    vit_model.eval()
    return vit_model

# Load CNN Model (ResNet18)
def load_cnn_model():
    cnn_model = models.resnet18(pretrained=False)
    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)  # Adjust output layer
    state_dict = torch.load("fake_real_cnn1.pth", map_location="cpu")

    # Fix key mismatch issue
    new_state_dict = {k.replace("cnn.", ""): v for k, v in state_dict.items()}  
    cnn_model.load_state_dict(new_state_dict)
    cnn_model.eval()
    return cnn_model

vit_model = load_vit_model()
cnn_model = load_cnn_model()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    image = Image.open(file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        vit_pred = vit_model(image_tensor).logits.argmax(-1).item()
        cnn_pred = cnn_model(image_tensor).argmax(-1).item()

    # Prediction Labels
    labels = ["REAL", "FAKE"]
    vit_result = labels[vit_pred]
    cnn_result = labels[cnn_pred]

    # Return JSON response
    return jsonify({'vit': vit_result, 'cnn': cnn_result})

if __name__ == '__main__':
    app.run(debug=True)
