import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Generate data
np.random.seed(42)
X = np.random.rand(100, 1) * 2  # 100 samples, 1 feature
y= 4 +3 * X + np.random.randn(100, 1)  # Linear relation with some noise

# Initialize parameters
n=100  # number of samples
theta = np.random.rand(2, 1)  # parameters (intercept and slope)

learning_rate = 0.01
iterations = 1000

# Add bias term to X
X_b = np.c_[np.ones((n, 1)), X]  # add bias term (intercept)

# Gradient Descent
for i in range(iterations):
    gradients = 2/n * X_b.T.dot(X_b.dot(theta) - y)  # compute gradients
    theta -= learning_rate * gradients  # update parameters

#prepare data 
X_tensor = tf.constant(X, dtype=tf.float32)
y_tensor = tf.constant(y, dtype=tf.float32)
#define model
class LinearRegressionModel(tf.Module):
    def __init__(self):
        self.W = tf.Variable(tf.random.normal([1, 1]), name='weight')
        self.b = tf.Variable(tf.random.normal([1]), name='bias')
    def __call__(self, x):
        return tf.matmul(x, self.W) + self.b
    
#loss function
def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

#Train with Stochastic Gradient Descent
model = LinearRegressionModel()
optimizer = tf.optimizers.SGD(learning_rate=0.01)
for epoch in range(1000):
    with tf.GradientTape() as tape:
        predictions = model(X_tensor)
        loss = mean_squared_error(y_tensor, predictions)
    gradients = tape.gradient(loss, [model.W, model.b])
    optimizer.apply_gradients(zip(gradients, [model.W, model.b]))
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")
        
import torch
import torch.nn as nn
import torch.optim as optim

# Prepare data
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
# Define model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # input size 1, output size 1
    def forward(self, x):
        return self.linear(x)
    
model = LinearRegressionModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()  # zero the parameter gradients
    outputs = model(X_tensor)  # forward pass
    loss = criterion(outputs, y_tensor)  # compute loss
    loss.backward()  # backward pass
    optimizer.step()  # update parameters
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")