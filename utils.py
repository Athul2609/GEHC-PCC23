import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pickle
import pytesseract
from PIL import Image
import re

def updater(line,final_explanation):
    words=line.split()
    start_index=words.index("feature")
    end_index=words.index("whose")
    attribute=" ".join(words[start_index+1:end_index])
    attribute=attribute[1:-1]
    if "less than or equal to" in line:
        comp="max"
    elif "greater than":
        comp="min"
    else:
        raise ValueError
    threshold_value=float(words[-1])
    if comp=="min":
        if(threshold_value<=final_explanation[(attribute,comp)]):
            final_explanation[(attribute,comp)]=threshold_value
    elif comp=="max":
        if(threshold_value>final_explanation[(attribute,comp)]):
            final_explanation[(attribute,comp)]=threshold_value
    return final_explanation

def explain_printer(final_explanation):
    result = ""
    reset = 1
    c = ""
    c1 = ""
    flag = 0
    for i, j in final_explanation.items():
        if reset == 0:
            if flag == 0:
                if j != float("-inf"):
                    # Format 'j' to have 4 decimal places
                    d = f"and {j:.4f}"
                    c += d
                    result += c
                    result += "\n"
                else:
                    result += c1
                    result += "\n"
            else:
                # Format 'j' to have 4 decimal places
                result += f"{i[0]} was less than or equal to {j:.4f}\n"
                flag = 0
            reset = 1
            continue
        if j != float("inf"):
            # Format 'j' to have 4 decimal places
            c = f"{i[0]} was in between {j:.4f} "
            c1 = f"{i[0]} was greater than {j:.4f}"
        else:
            flag = 1
        reset = 0
    return result

def generate_trees():
    with open('model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    attribute_list = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']
    class_names = ['breast cancer free', 'breast cancer']
    
    for idx, clf in enumerate(loaded_model.estimators_):
        plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
        plot_tree(clf, filled=True, feature_names=attribute_list, class_names=class_names)

        # Save each tree with a unique name
        plt.savefig(f'static/decision_tree_{idx}.png')

def extract_attributes_from_image(image_path):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    try:
        # Open the image using PIL (Python Imaging Library)
        img = Image.open(image_path)

        # Use Tesseract OCR to extract text from the image
        extracted_text = pytesseract.image_to_string(img)

        # Define a list of attribute names to search for
        attribute_list = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
            'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
            'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
            'area error', 'smoothness error', 'compactness error', 'concavity error',
            'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius',
            'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
            'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry',
            'worst fractal dimension'
        ]

        attribute_values = {}
        attr = [0.0] * 30  # Initialize with zeros or appropriate default values
        for attribute in attribute_list:
            # Search for the attribute name and its corresponding value in the extracted text
            pattern = re.compile(rf"{attribute}: ([\d.]+)")
            match = re.search(pattern, extracted_text)
            if match:
                value_str = match.group(1)
                if value_str:
                    attribute_values[attribute] = float(value_str)
                else:
                    attribute_values[attribute] = None
            else:
                attribute_values[attribute] = None

        for attribute, value in attribute_values.items():
            if value is not None:
                attr[attribute_list.index(attribute)] = value

        return attr

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


if __name__ == "__main__":
    image_path = r'C:\Users\athul\myfiles\competitions\GEHC precision care challenge\finals\mock patient report.png'  # Replace with the path to your image
    attributes = extract_attributes_from_image(image_path)
    if attributes:
        print(attributes)