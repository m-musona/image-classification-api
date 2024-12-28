import torch
from torchvision import models, transforms
from PIL import Image
import json

model = None
transform = None

def init():
    # Load the pretrained ResNet-50 model
    global model
    model = models.resnet50(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    print("Model Created")

    # Define the image transformation pipeline
    global transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing the image to 224x224
        transforms.ToTensor(),         # Convert the image to a tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Normalizing with ImageNet means
            std=[0.229, 0.224, 0.225]
        )
    ])
    print("transform Created")

# Load and preprocess the image
def preprocess_image(image_path):
    print("Starting Preprocessing Image")
    image = Image.open(image_path).convert("RGB")  # Open the image and convert it to RGB
    print("Finished Preprocessing Image")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Classify the image
def classify_image(image_path):
    print("Starting Classifying Image")
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)  # Perform inference
    probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Convert logits to probabilities
    print("Performed Inference")
    
    # Load the labels for ImageNet classes
    with open("imagenet_classes.json") as f:
        labels = json.load(f)
    print("Loaded Classes")

    # Get the top 5 predictions
    print("Getting Top 5 Predictions")
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    results = {}
    for i in range(top5_prob.size(0)):
        results[labels[top5_catid[i]]] = round(top5_prob[i].item(), 2)
    return results

# Example usage
# image_path = "1.jpeg"  # Replace with the path to your image
# classify_image(image_path)
