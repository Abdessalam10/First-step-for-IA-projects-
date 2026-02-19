import numpy as np

#mean squared error loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#Binary cross-entropy loss function
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clip predictions to avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

#Example usage
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.7])
mse_loss = mean_squared_error(y_true, y_pred)
bce_loss = binary_cross_entropy(y_true, y_pred)
print(f"Mean Squared Error Loss: {mse_loss:.8f}")
print(f"Binary Cross-Entropy Loss: {bce_loss:.8f}")

#Derivative of mean squared error loss with respect to predictions
def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)
#Derivative of binary cross-entropy loss with respect to predictions
def bce_derivative(y_true, y_pred):
    epsilon = 1e-15  # to avoid division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clip predictions to avoid division by zero
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * len(y_true))
#Example usage of derivatives
mse_grad = mse_derivative(y_true, y_pred)
bce_grad = bce_derivative(y_true, y_pred)
print(f"Gradient of MSE Loss: {mse_grad}")
print(f"Gradient of Binary Cross-Entropy Loss: {bce_grad}")

#Note: In a real training loop, these gradients would be used to update the model's weights using an optimization algorithm like gradient descent.
np.random.seed(42)
n_samples = 100
n_features = 10
X = np.random.rand(n_samples, n_features)
y_true = np.random.randint(0, 2, size=(n_samples,))
y_pred = np.random.rand(n_samples)
mse_loss = mean_squared_error(y_true, y_pred)
bce_loss = binary_cross_entropy(y_true, y_pred)
print(f"Mean Squared Error Loss on Random Data: {mse_loss:.8f}")
print(f"Binary Cross-Entropy Loss on Random Data: {bce_loss:.8f}")

