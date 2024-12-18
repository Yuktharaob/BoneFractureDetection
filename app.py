from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from torchvision import models, transforms
import os

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture and weights
model_path = "bone_fracture_resnet18.pth"

# Check if the model path exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at '{model_path}'.")

# Initialize the ResNet18 model with no pretrained weights
model = models.resnet18(weights=None)  # Use `weights` instead of `pretrained`
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary classification: fractured, not fractured

# Load model weights securely
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Class names
classes = ["fractured", "not fractured"]

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if not file:
        return jsonify({"error": "No file selected"})

    try:
        # Process the uploaded image
        image = Image.open(file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():  # Ensure gradients are not computed
            outputs = model(input_tensor)  # Model prediction
            probabilities = torch.softmax(outputs, dim=1).cpu().detach()  # Detach tensor
            predicted_class = probabilities.argmax(1).item()
            confidence = probabilities[0][predicted_class].item()

        prediction = classes[predicted_class]
        return jsonify({"prediction": prediction, "confidence": round(confidence * 100, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
