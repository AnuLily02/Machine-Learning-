
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to evaluate the model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    print("LinearRegression coefficients:", model.coef_)
    print("LinearRegression intercept:", model.intercept_)
    print("LinearRegression training score (R^2):", model.score(X_train, y_train))
    print("LinearRegression test score (R^2):", model.score(X_test, y_test))
    
    train_predictions = model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    mae = mean_absolute_error(y_train, train_predictions)
    print("LinearRegression RMSE:", rmse)
    print("LinearRegression MAE:", mae)
    print("#####################################")

# Linear Regression
lin_reg = LinearRegression()
evaluate_model(lin_reg, X_train, y_train, X_test, y_test)
