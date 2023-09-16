from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from explainer import explain_predict
from sklearn.tree import export_graphviz

import pickle

data = load_breast_cancer()
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],  
    'max_depth': [None, 10, 20, 30], 
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4]   
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

best_rf_classifier = grid_search.best_estimator_

with open('model.pkl', 'wb') as file:
    pickle.dump(best_rf_classifier, file)

y_pred = best_rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Best Model with original none explainable inference: {accuracy:.2f}")

with open("output.txt", "w") as output_file:
    y_pred=explain_predict(X_test,best_rf_classifier,output_file)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Best Model with explainable inference: {accuracy:.2f}")

 # This will open an interactive visualization in your default web browser





