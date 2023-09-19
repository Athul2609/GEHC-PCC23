import torch
from torch import nn, optim
import os
from efficientnet_pytorch import EfficientNet
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # so this is basically like use gpu if you can if the system doesn't have a gpu use cpu
LEARNING_RATE = 3e-5 # basic learning rate onlu
WEIGHT_DECAY = 5e-4 # so this is a parameter used in L2 regularisation which is used to reduce overfitting
CHECKPOINT_FILE = r"C:\Users\athul\myfiles\competitions\GEHC precision care challenge\finals\b3.pth.tar"
LOAD_MODEL = True 

val_transforms = A.Compose([
    A.Resize(height=760, width=760),
    A.Normalize(
        mean=[0.3199, 0.2240, 0.1609],
        std=[0.3020, 0.2183, 0.1741],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])

def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# Function to perform inference on a single image
def predict_single_image(image_path, model, transform):
    image = Image.open(image_path)
    image = np.array(image)

    if transform:
        image = transform(image=image)["image"]

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        model.eval()
        prediction = model(image)

    return prediction

def load_base_model():
    # Load the model and checkpoint
    model = EfficientNet.from_pretrained("efficientnet-b3")
    model._fc = nn.Linear(1536, 1)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if LOAD_MODEL and os.path.isfile(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=torch.device(DEVICE))
        load_checkpoint(checkpoint, model, optimizer, LEARNING_RATE)
    
    return model

def predict_one_eye(image_path,model):
# Example usage: Predict on an image path provided by the user
    user_image_path =  image_path # Replace with the user's image path
    prediction = predict_single_image(user_image_path, model, val_transforms)
    print("Raw Prediction:", prediction.item())
    raw_pred=prediction.item()
    prediction=float(prediction.item())
    final_pred=prediction//1
    if prediction%1>0.5:
        final_pred+=1

    # Print the prediction
    print("Cleaned Prediction:", final_pred)
    return raw_pred,final_pred

# image should be passed left then right
def get_blend(imgs, model):
    model.eval()

    images = imgs.to(DEVICE)

    with torch.no_grad():
        features = F.adaptive_avg_pool2d(
            model.extract_features(images), output_size=1
        )
        features_logits = features.reshape(features.shape[0] // 2, 2, features.shape[1])
        preds = model(images).reshape(images.shape[0] // 2, 2, 1)
        new_features = (
            torch.cat([features_logits, preds], dim=2)
            .view(preds.shape[0], -1)
            .cpu()
            .numpy()
        )
    return new_features

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d((1536 + 1) * 2),
            nn.Linear((1536+1) * 2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        return self.model(x)
    
def load_second_model():
    # Define the path to your saved model file
    model_path = 'linear.pth.tar'

    # Create an instance of your MyModel class
    model = MyModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    if LOAD_MODEL and os.path.isfile(r"C:\Users\athul\myfiles\competitions\GEHC precision care challenge\finals\linear.pth.tar"):
        checkpoint = torch.load(r"C:\Users\athul\myfiles\competitions\GEHC precision care challenge\finals\linear.pth.tar", map_location=torch.device(DEVICE))
        load_checkpoint(checkpoint, model, optimizer, lr=1e-4)
    
    return model


def make_prediction(model, input):
    preds = []
    model.eval()
    input = torch.from_numpy(input)
    input = input.to(DEVICE)

    with torch.no_grad():
        predictions = model(input)
        # Convert MSE floats to integer predictions
        predictions[predictions < 0.5] = 0
        predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
        predictions[(predictions >= 1.5) & (predictions < 2.5)] = 2
        predictions[(predictions >= 2.5) & (predictions < 3.5)] = 3
        predictions[(predictions >= 3.5) & (predictions < 1000000000000)] = 4
        predictions = predictions.long().view(-1)

        preds.append(predictions.cpu().numpy())
    return preds

def final_dr_pred(left_eye_path,right_eye_path):
    left_eye_image = Image.open(left_eye_path)
    right_eye_image = Image.open(right_eye_path)
    left_eye_image = np.array(left_eye_image)
    right_eye_image = np.array(right_eye_image)


    left_eye_image = val_transforms(image=left_eye_image)["image"]
    right_eye_image = val_transforms(image=right_eye_image)["image"]

    left_eye_image = left_eye_image.unsqueeze(0).to(DEVICE)
    right_eye_image = right_eye_image.unsqueeze(0).to(DEVICE)

    # Combine the left and right eye tensors into a single input tensor
    input_tensor = torch.cat((left_eye_image, right_eye_image), dim=0)

    base_model=load_base_model()
    second_model=load_second_model()
    feauture_input=get_blend(input_tensor, base_model)
    return make_prediction(second_model, feauture_input)


if __name__=='__main__':
    # Directory containing the images
    dir_path = r"C:\Users\athul\myfiles\Research\QML\train\images_resized_1000"

    # Input patient ID
    patient_id = input("Enter patient ID: ")

    # Define the file paths for the left and right eye images
    left_eye_path = os.path.join(dir_path, f"{patient_id}_left.jpeg")
    right_eye_path = os.path.join(dir_path, f"{patient_id}_right.jpeg")

    pred=final_dr_pred(left_eye_path,right_eye_path)
    print(pred)

    import sys
    sys.exit()

    

    left_eye_image = Image.open(left_eye_path)
    right_eye_image = Image.open(right_eye_path)
    left_eye_image = np.array(left_eye_image)
    right_eye_image = np.array(right_eye_image)


    left_eye_image = val_transforms(image=left_eye_image)["image"]
    right_eye_image = val_transforms(image=right_eye_image)["image"]

    left_eye_image = left_eye_image.unsqueeze(0).to(DEVICE)
    right_eye_image = right_eye_image.unsqueeze(0).to(DEVICE)

    print(left_eye_image.shape)
    # Combine the left and right eye tensors into a single input tensor
    input_tensor = torch.cat((left_eye_image, right_eye_image), dim=0)

    # Convert the input tensor to a numpy array
    # input_numpy = input_tensor.numpy()

    # Print the shape of the input numpy array
    print("Input Numpy Array Shape:", input_tensor.shape)
    base_model=load_base_model()
    second_model=load_second_model()
    print(predict_one_eye(left_eye_path,base_model))
    print(predict_one_eye(right_eye_path,base_model))
    feauture_input=get_blend(input_tensor, base_model)
    print(type(feauture_input[0]))
    print(feauture_input.shape)
    print(make_prediction(second_model, feauture_input))
