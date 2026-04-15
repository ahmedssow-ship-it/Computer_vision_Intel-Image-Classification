
import argparse
import numpy as np
from PIL import Image

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


# ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--framework', type=str, required=True,
                    choices=['pytorch', 'tensorflow'])
parser.add_argument('--image_path', type=str, required=True)

args = parser.parse_args()


# PREPROCESSING
def preprocess_image_pytorch(image_path):
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def preprocess_image_tf(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image



# PYTORCH
if args.framework == "pytorch":
    import torch
    from models.model_pytorch import IntelCNNPyTorch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IntelCNNPyTorch()
    model.load_state_dict(torch.load("ahmed_model.pth", map_location=device))
    model.to(device)
    model.eval()

    image = preprocess_image_pytorch(args.image_path).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    print("Predicted class:", CLASSES[predicted.item()])


# TENSORFLOW
elif args.framework == "tensorflow":
    import tensorflow as tf

    model = tf.keras.models.load_model("ahmed_model.keras")

    image = preprocess_image_tf(args.image_path)

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)

    print("Predicted class:", CLASSES[predicted_class])
    
