import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
#Load the California housing dataset
from sklearn.datasets import fetch_california_housing
# Load the California housing dataset
california = fetch_california_housing(as_frame=True)
df = california.frame

# Selecting features and target variable
X = df[['MedInc']]
y= df['MedHouseVal']

# Transforming the data
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

#Fit polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)
#Make predictions
y_pred = model.predict(X_poly)

# Plot actual s vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data Points', alpha=0.5)
plt.scatter(X, y_pred, color='red', label='Predicted Data Points', alpha=0.5)

plt.title('Polynomial Regression Fit on California Housing Data')
plt.xlabel('Median Income (MedInc)')
plt.ylabel('Median House Value (MedHouseVal)')
plt.legend()
plt.show()
# Evaluate performance
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
print("Ridge Regression - Mean Squared Error:", ridge_mse)
print("Ridge Regression - R^2 Score:", ridge_r2)
# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)
print("Lasso Regression - Mean Squared Error:", lasso_mse)
print("Lasso Regression - R^2 Score:", lasso_r2)

# Visualize Ridge and Lasso predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual Data Points', alpha=0.5)
plt.scatter(X_test[:, 0], ridge_pred, color='red', label='Ridge Predicted Data Points', alpha=0.5)
plt.scatter(X_test[:, 0], lasso_pred, color='green', label='Lasso Predicted Data Points', alpha=0.5)
plt.xlabel('Median Income (MedInc)')
plt.ylabel('Median House Value (MedHouseVal)')
plt.title('Comparison of Ridge and Lasso Predictions')
plt.legend()
plt.show()