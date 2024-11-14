import pandas as pd

import streamlit as st

# Import necessary libraries for various classification models
# Linear Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# Support Vector Machines
from sklearn.svm import SVC

# Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Tree-Based Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Ensemble Methods
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

# Neural Networks
from sklearn.neural_network import MLPClassifier

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB

# Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Specialized Libraries
from catboost import CatBoostClassifier

# Import metrics for evaluation
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from sklearn.multioutput import MultiOutputClassifier

# Initialize a list of classification models with default or specified hyperparameters
models = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs'),
    RidgeClassifier(),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.0001, solver='adam', random_state=42),
    MultinomialNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    CatBoostClassifier()
]

# Function to perform classification using multiple models and hyperparameter tuning
def Classification(X_train, X_test, y_train, y_test, models=models):
    """
    Function to perform classification using multiple models with hyperparameter tuning
    Args:
    X_train: Training data features
    y_train: Training data labels
    X_test: Test data features
    y_test: Test data labels
    models: List of models to be used for classification
    param_grids: Parameter grids for hyperparameter tuning
    
    Returns:
    Prints the evaluation metrics for each model
    """
    # Loop through each model, fit on training data, predict on test data, and compute metrics

    acc = []

    for i, model in enumerate(models):
        
        # Initialize MultiOutputClassifier with the current model
        multi_output_classifier = MultiOutputClassifier(model)

        # Fit the model on the training data
        multi_output_classifier.fit(X_train, y_train)
        
        # Predict on the test data
        y_pred = multi_output_classifier.predict(X_test)

        acc.append(accuracy_score(y_test , y_pred))
        
        # Print model evaluation results
        st.write("\n------------------------------------------------------------------------------------\n")
        st.write("Model:", model.__class__.__name__)
        st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        st.write("Precision Score:", precision_score(y_test, y_pred, average='weighted'))
        st.write("Recall Score:", recall_score(y_test, y_pred, average='weighted'))
        st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
        st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
        st.write("\n------------------------------------------------------------------------------------")

    return pd.DataFrame({
        'models' : models,
        'accuracy_score' : acc,
    })
