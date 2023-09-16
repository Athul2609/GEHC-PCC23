import numpy as np

def traverse_tree(node, X, feature_names,tree,output_file):
    if tree.children_left[node] == tree.children_right[node]:
        if(tree.value[node][0][1]>tree.value[node][0][0]):
            c="Tree has come to the conclusion that patient has breast cancer"
        else:
            c="Tree has come to the conclusion that patient does not have breast cancer "
        output_file.write(f"Leaf Node {c} \n")
        return tree.value[node]
    
    feature_index = tree.feature[node]
    threshold = tree.threshold[node]

    feature_name = feature_names[feature_index]
    feature_value = X[0, feature_index]

    output_file.write(f"Node: We are examining feature '{feature_name}', whose value {feature_value}, which will be compared to the node threshold {threshold}\n")
    
    if feature_value <= threshold:
        output_file.write(f"Going left: as feature '{feature_name}' whose value is {feature_value} is less than or equal to node threshold value, {threshold}\n")
        return traverse_tree(tree.children_left[node], X, feature_names,tree,output_file)
    else:
        output_file.write(f"Going right: as feature '{feature_name}' whose value is {feature_value} is greater than node threshold value, {threshold}\n")
        return traverse_tree(tree.children_right[node], X, feature_names,tree,output_file)

def explain_predict(X_test,best_rf_classifier,output_file,feature_names = ['mean radius',
 'mean texture',
 'mean perimeter',
 'mean area',
 'mean smoothness',
 'mean compactness',
 'mean concavity',
 'mean concave points',
 'mean symmetry',
 'mean fractal dimension',
 'radius error',
 'texture error',
 'perimeter error',
 'area error',
 'smoothness error',
 'compactness error',
 'concavity error',
 'concave points error',
 'symmetry error',
 'fractal dimension error',
 'worst radius',
 'worst texture',
 'worst perimeter',
 'worst area',
 'worst smoothness',
 'worst compactness',
 'worst concavity',
 'worst concave points',
 'worst symmetry',
 'worst fractal dimension']):
        X_pred = []

        for i in X_test:
            i = i.reshape(1, -1)
            preds = np.array([0, 0])
            output_file.write(f"\nInput: {i}\n")
            count=1
            for j in best_rf_classifier.estimators_:
                output_file.write(f"\ntree no.{count}\n")
                prediction = traverse_tree(0, i, feature_names, j.tree_,output_file)
                prediction = np.array(prediction)
                prediction = prediction.reshape(-1)
                prediction = np.argmax(prediction)
                preds[prediction] += 1
                count+=1
            if np.argmax(preds) == 0:
                c=f"\n{preds[0]} trees came to the conclusion that the patient doesn't have Breast Cancer while only {preds[1]} trees came to the conclusion that the patient has Breast Cancer, so the model has come to the conclusion that the patient doesn't have Breast Cancer\n"
            else:
                c=f"\n{preds[1]} trees came to the conclusion that the patient has Breast Cancer while only {preds[0]} trees came to the conclusion that the patient doesn't have Breast Cancer, so the model has come to the conclusion that the patient has Breast Cancer\n"
            output_file.write(c)
            X_pred.append(np.argmax(preds))

        return X_pred