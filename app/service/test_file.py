import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import jsonify
from core.load_models.resnet_v2 import model2, class_names



def test_file_service(file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define transformations to be applied to the image
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])

    # Load the image
    image_path = file
    image = Image.open(image_path).convert("RGB")

    # Apply transformations to the image
    image = data_transform(image)

    # Add batch dimension and move the image to the GPU if available
    image = image.unsqueeze(0).to(device)

    model2.to(device)

    model2.eval()
    # Perform prediction
    with torch.inference_mode():
        outputs = model2(image)
        print(outputs)
        predicted = outputs.argmax(dim=1)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Map the predicted label index to its corresponding class name
    predicted_class = class_names[predicted.item()]

    print("Predicted Class:", predicted_class)

    # Get top-5 predictions along with their confidence
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(f"Category: {top5_catid[i]}, Probability: {top5_prob[i].item()}")

    return jsonify({'status': 'Success', 'Class name': predicted_class, 'Category': (str)(top5_catid[0].item()),'accuracy': (str)(top5_prob[0].item())[0:5]})