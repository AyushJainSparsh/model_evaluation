import pandas as pd
import numpy as np

# Import necessary libraries for various regression models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score , root_mean_squared_error
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
# Import necessary libraries for various classification models
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier

from sklearn.multioutput import MultiOutputClassifier , MultiOutputRegressor

# Initialize a list of regression models with default or specified hyperparameters
models_reg = [
    LinearRegression(),
    Lasso(),
    Ridge(),
    ElasticNet(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    AdaBoostRegressor(),
    SVR(),
    KNeighborsRegressor(),
    CatBoostRegressor(depth=10, learning_rate=0.1, iterations=100)
]

models_cla = [
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

def Regression_best_model(result , X_train , y_train):
    max = result['r2_score'].idxmax()
    model = models_reg[max].fit(X_train , y_train)

    return model

def Classification_best_model(result , X_train , y_train):
    max = result['accuracy_score'].idxmax()
    model = models_cla[max].fit(X_train , y_train)

    return model

def Multi_Regression_best_model(result , X_train , y_train):
    max = result['r2_score'].idxmax()
    model = MultiOutputRegressor(models_reg[max])
    model = model.fit(X_train , y_train)

    return model

def Multi_Classification_best_model(result , X_train , y_train):
    max = result['accuracy_score'].idxmax()
    model = MultiOutputClassifier(models_cla[max])
    model = model.fit(X_train , y_train)

    return model