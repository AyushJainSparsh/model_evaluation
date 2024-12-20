import pandas as pd

# Import necessary libraries for various regression models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score , root_mean_squared_error
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

# Initialize a list of regression models with default or specified hyperparameters
models = [
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

# Define the regression function
def Regression(X_train, X_test, y_train, y_test, models=models):
    """
    Function to perform multi-output regression using various models with hyperparameter tuning
    Args:
    X_train: Training data features
    y_train: Training data labels
    X_test: Test data features
    y_test: Test data labels
    models: List of models to be used for regression
    param_grids: Parameter grids for hyperparameter tuning
    
    Returns:
    A DataFrame containing the evaluation metrics for each model
    """
    # Initialize lists to store evaluation metrics
    mse = []
    mae = []
    rmse = []
    score = []
    
    # Loop through each model, fit on training data, predict on test data, and compute metrics
    for i in range(len(models)):
        # Initialize MultiOutputRegressor with the current model
        multi_output_regressor = MultiOutputRegressor(models[i])
        
        # Initialize GridSearchCV with the current model and its parameter grid
        multi_output_regressor.fit(X_train , y_train)
        
        # Predict on the test data
        y_pred = multi_output_regressor.predict(X_test)
        
        # Compute and store the evaluation metrics
        mse.append(mean_squared_error(y_test, y_pred))
        mae.append(mean_absolute_error(y_test, y_pred))
        rmse.append(root_mean_squared_error(y_test, y_pred)) 
        score.append(r2_score(y_test, y_pred))
    
    # Create a DataFrame to store and display the results
    output = {
        "Model": [model.__class__.__name__ for model in models],
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "r2_score": score,
    }
    
    return pd.DataFrame(output)

# Example usage:
# X_train, X_test, y_train, y_test = ... # Define your data
# results = RegressionGrid(X_train, y_train, X_test, y_test)
# print(results)
