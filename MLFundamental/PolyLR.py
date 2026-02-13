import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generating synthetic data
np.random.seed(42)
X = 100 * np.random.rand(100, 1)
y = 4 + 3 * X + 0.5 * X**2 + np.random.randn(100, 1) * 100

# Transforming data to include polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly_features.fit_transform(X)

# Fit Linear Regression Model on polynomial features
model = LinearRegression()
model.fit(X_poly, y)
# Make Prediction
y_pred = model.predict(X_poly)

# Print Coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
# Plotting the results
plt.scatter(X, y, color='blue', label='Actual Data Points')
plt.scatter(X, y_pred, color='red', label='Predicted Data Points')
plt.title('Polynomial Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
# Evaluate performance
mean_squared_error = mean_squared_error(y, y_pred)  
r2_score = r2_score(y, y_pred)
print("Mean Squared Error:", mean_squared_error)
print("R2 Score:", r2_score)