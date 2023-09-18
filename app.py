import pickle
from flask import Flask, render_template, request, jsonify
from explainer import explain_predict
from cleaner import cleaner
import numpy as np
import os
from utils import extract_attributes_from_image
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from PIL import Image  # Import the Python Imaging Library
import argparse
import time
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
from efficientnet_pytorch import EfficientNet
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import pandas as pd
from PIL import Image

import torch
from torchvision.models import vgg16, VGG16_Weights

from src.data import get_data_loader
from src.lrp import LRPModel
app = Flask(__name__)

INPUT_FOLDER = 'input/cats'
OUTPUT_FOLDER = 'static'
app.config['INPUT_FOLDER'] = INPUT_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load the machine learning model from the pickle file
with open('model.pkl', 'rb') as file:
    print(type(file))
    loaded_model = pickle.load(file)


res_dic={0:"Patient doesn't have Breast Cancer",1:"Patient has Breast Cancer"}

# Define the attribute names corresponding to the input features
attribute_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
    'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error',
    'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

@app.route('/model', methods=['GET', 'POST'])
def breast_cancer_detection():
    if request.method == 'POST':
        # Collect user input for each attribute and create a list
        X_test = []
        for attribute_name in attribute_names:
            value = float(request.form[attribute_name])
            X_test.append(value)

        X_test=np.array(X_test)
        with open("sample.txt", "w") as output_file:
            y_pred = explain_predict([X_test], loaded_model, output_file)
        y_pred=res_dic[int(y_pred[0])]
        result = cleaner("sample.txt")

        explanation_items = result.split('\n')[:-1]
        last_line= result.split('\n')[-1]

        # Assign the list of explanation items to 'explanation_items' and the last line to 'last_line'
        # explanation_items, last_line = explanation_items if len(explanation_items) > 1 else ([], result)

        # Pass both 'explanation_items' and 'last_line' to the template
        return render_template('result.html', y_pred=y_pred, explanation_items=explanation_items, last_line=last_line)
    
    return render_template('model.html', attribute_names=attribute_names)

@app.route('/about_trees')
def about():
    # Get a list of image filenames that start with "decision_tree"
    image_list = [filename for filename in os.listdir('static') if filename.startswith('decision_tree')]
    
    return render_template('about_trees.html', image_list=image_list)

@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/process_image', methods=['GET', 'POST'])
def process_image():
    if request.method == 'POST':
        try:
            # Ensure that an image file is uploaded
            if 'image' not in request.files:
                return jsonify({"error": "No image file provided"}), 400

            # Get the uploaded image file
            image = request.files['image']

            # Check if the file is not empty
            if image.filename == '':
                return jsonify({"error": "Empty image file"}), 400

            # Check if the file is an image
            if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                return jsonify({"error": "Unsupported image format"}), 400

            # Save the uploaded image temporarily
            temp_image_path = "temp_image.png"
            image.save(temp_image_path)

            X_test=extract_attributes_from_image(temp_image_path)

            # Clean up the temporary image file
            

            X_test=np.array(X_test)
            with open("sample.txt", "w") as output_file:
                y_pred = explain_predict([X_test], loaded_model, output_file)
            y_pred=res_dic[int(y_pred[0])]
            result = cleaner("sample.txt")

            explanation_items = result.split('\n')[:-1]
            last_line= result.split('\n')[-1]

            # Assign the list of explanation items to 'explanation_items' and the last line to 'last_line'
            # explanation_items, last_line = explanation_items if len(explanation_items) > 1 else ([], result)
            os.remove(temp_image_path)
            # Pass both 'explanation_items' and 'last_line' to the template
            return render_template('result.html', y_pred=y_pred, explanation_items=explanation_items, last_line=last_line)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Render the HTML page for uploading an image (GET request)
    return render_template('ocr_input.html')

@app.route('/lrp')
def lrp():
    return render_template('lrp.html', output_image=None)

@app.route('/blog')
def blog():
    return render_template('blog.html', output_image=None)

@app.route('/upload', methods=['POST'])
def upload():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        help="Input directory.",
        default="./input/",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Output directory.",
        default="./output/",
    )
    parser.add_argument(
        "-b", "--batch-size", dest="batch_size", help="Batch size.", default=1, type=int
    )
    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        help="Device.",
        choices=["gpu", "cpu"],
        default="gpu",
        type=str,
    )
    parser.add_argument(
        "-k",
        "--top-k",
        dest="top_k",
        help="Proportion of relevance scores that are allowed to pass.",
        default=0.02,
        type=float,
    )
    parser.add_argument(
        "-r",
        "--resize",
        dest="resize",
        help="Resize image before processing.",
        default=0,
        type=int,
    )

    config = parser.parse_args()

    pathlib.Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    if 'file' not in request.files:
        return redirect(url_for('lrp'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('lrp'))

    if file:
        # Save the uploaded file to the INPUT_FOLDER
        input_folder_list = os.listdir(app.config['INPUT_FOLDER'])
        if len(input_folder_list) > 1:
            print(len(input_folder_list))
            for i in range(len(input_folder_list)):
                os.remove(os.path.join("input/cats",input_folder_list[i]))
            print(file.filename)
        input_filename = os.path.join(app.config['INPUT_FOLDER'], file.filename)
        file.save(input_filename)

        # Perform transformations on the image (example: resizing)
        transformed_image = per_image_lrp(config)

        # Save the transformed image to the OUTPUT_FOLDER
        transformed_image = transformed_image.detach().cpu().numpy()
        original_image = plt.imread(os.path.join(app.config['INPUT_FOLDER'], file.filename))
        plt.imsave(os.path.join(app.config['OUTPUT_FOLDER'], 'original_image.png'), original_image)
        plt.imsave(os.path.join(app.config['OUTPUT_FOLDER'], 'transformed_image.png'), transformed_image, cmap='afmhot')
        images = []
        output_folder = 'static'
        for filename in os.listdir(output_folder):
            if filename=='original_image.png':
                images.append(os.path.join(output_folder, filename))
        for filename in os.listdir(output_folder):
            if filename=='transformed_image.png':
                images.append(os.path.join(output_folder, filename))
        print(images)
        
        return render_template('lrp.html', images=images)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

def per_image_lrp(config: argparse.Namespace) -> None:
    """Test function that plots heatmaps for images placed in the input folder.

    Images have to be placed in their corresponding class folders.

    Args:
        config: Argparse namespace object.

    """
    if config.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using: {device}\n")

    data_loader = get_data_loader(config)

    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model.to(device)

    lrp_model = LRPModel(model=model, top_k=config.top_k)

    for i, (x, y) in enumerate(data_loader):
        x = x.to(device)
        # y = y.to(device)  # here not used as method is unsupervised.

        t0 = time.time()
        r = lrp_model.forward(x)
        print("{time:.2f} FPS".format(time=(1.0 / (time.time() - t0))))

    return r

if __name__ == '__main__':
    app.run(debug=True)