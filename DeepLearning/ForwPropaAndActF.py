import numpy as np
import matplotlib.pyplot as plt
# DEfine the activation function (ReLU)
def relu(x):
    return np.maximum(0, x)
#Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#Define tanh activation function
def tanh(x):
    return np.tanh(x)
#Define softmax activation function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
# Define the forward propagation function
def forward_propagation(X, W1, b1, W2, b2 ):
    # Layer 1
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    
    # Layer 2 (Output Layer)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return a2

#Example inputs
X = np.array([[0.5,0.7], [0.8,0.2]])
weights1 = np.array([[0.2, 0.4], [0.6, 0.8]])
biases1 = np.array([0.1, 0.2])

weights2 = np.array([[0.3, 0.5], [0.7, 0.9]])
biases2 = np.array([0.1, 0.2])

#Perform forward propagation
output = forward_propagation(X, weights1, biases1, weights2, biases2)
print("Output of the network:\n", output)

#define range of input values for visualization
x = np.linspace(-10, 10, 100)
#Visualize activation functions
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(x, relu(x))
plt.title("ReLU Activation Function")
plt.subplot(2, 2, 2)
plt.plot(x, sigmoid(x))
plt.title("Sigmoid Activation Function")
plt.subplot(2, 2, 3)
plt.plot(x, tanh(x))
plt.title("Tanh Activation Function")
plt.subplot(2, 2, 4)
plt.plot(x, softmax(np.array([x, x]))[0])  # Softmax applied to two identical inputs for visualization
plt.title("Softmax Activation Function")
plt.tight_layout()
plt.show()