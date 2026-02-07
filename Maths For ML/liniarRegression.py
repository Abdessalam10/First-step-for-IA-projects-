import numpy as np

np.random.seed(42)
# Generate synthetic data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# Add bias term to X
X_b = np.c_[np.ones((100, 1)), X]
# Initialize parameters
theta = np.random.randn(2, 1)
# Hyperparameters
learning_rate = 0.01
n_iterations = 1000
# Train the model using Gradient Descent
theta_optimal = gradient_descent(X_b, y, theta, learning_rate, n_iterations)
# Evaluate the model
y_pred = predict(X_b, theta_optimal)
mse = mean_squared_error(y, y_pred)
r2 = r_squared(y, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

print("Optimal parameters (theta):", theta_optimal)
def predict(X, theta):
    return X @ theta
def gradient_descent(X, y, theta, learning_rate, n_iterations):
    m = len(y)
    for iteration in range(n_iterations):
        gradients = 1/m * X.T @ (predict(X, theta) - y)
        theta = theta - learning_rate * gradients
    return theta

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)