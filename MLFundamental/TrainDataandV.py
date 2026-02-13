import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error ,r2_score

#Generating synthetic data
np.random.seed(42)
X = 100 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

#Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Fit Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

#make Prediction
y_pred = model.predict(X_test)
#Print Coefficients
print("Coefficients:", model.coef_[0][0])
print("Intercept:", model.intercept_[0])


plt.scatter(X_test, y_test, color='blue', label='Test Data Points')
plt.plot(X_test, y_pred, color='red', label='Predicted Values')

plt.title('Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Ealuate performance
mean_squared_error = mean_squared_error(y_test, y_pred)
r2_score = r2_score(y_test, y_pred)

print("Mean Squared Error:", mean_squared_error)
print("R^2 Score:", r2_score)