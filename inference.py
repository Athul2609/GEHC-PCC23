import pickle
from explainer import explain_predict
from cleaner import cleaner

with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

attribute_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
    'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error',
    'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

X_test = []
for attribute_name in attribute_names:
    value = float(input(f"Enter the value for {attribute_name}: "))
    X_test.append(value)

with open("sample.txt", "w") as output_file:
    y_pred = explain_predict(X_test, loaded_model, output_file)

result = cleaner("sample.txt")

print(result)
