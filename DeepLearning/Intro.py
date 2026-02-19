from tensorflow.keras.datasets import mnist , cifar10
import tensorflow as tf
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# Load MNIST dataset
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
print("MNIST Dataset:")
print("Training set shape:", x_train_mnist.shape)
print("Test set shape:", x_test_mnist.shape)

# Load CIFAR-10 dataset
(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()
print("\nCIFAR-10 Dataset:")
print("Training set shape:", x_train_cifar.shape)
print("Test set shape:", x_test_cifar.shape)

#Define a basic  dense layer
layer = tf.keras.layers.Dense(128, activation='relu')
print("\nDense Layer: ", layer)

#Ddefine a basic dense layer
layer = nn.Linear(in_features=10, out_features=5)
print("\nPyTorch Linear Layer: ", layer)

#Visualize some samples from MNIST dataset
plt.figure(figsize=(10, 4))
plt.imshow(x_train_mnist[0], cmap='gray')
plt.title(f"Label: {y_train_mnist[0]}")
plt.show

#Visualize some samples from CIFAR-10 dataset
plt.figure(figsize=(10, 4))
plt.imshow(x_train_cifar[0])
plt.title(f"Label: {y_train_cifar[0][0]}")
plt.show()