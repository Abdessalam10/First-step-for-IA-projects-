import sympy as sp
import numpy as np
#define the variable and the function function
x = sp.symbols('x')
f = sp.exp(-x)

#compute indefinite integral
indefinite_integral = sp.integrate(f, x)
print("Indefinite Integral:", indefinite_integral)
#compute definite integral from 0 to infinity
definite_integral = sp.integrate(f, (x, 0, sp.oo))
print("Definite Integral from 0 to infinity:", definite_integral)

#  θ = θ - η * [ 2 * xi.T * (xi * θ - yi) ]

np.random.seed(42)
X= 2 * np.random.rand(100, 1)
y= 4 + 3 * X + np.random.randn(100, 1)

# Add bias term to X
X_b = np.c_[np.ones((100, 1)), X]

# DGD Implementation

def stochastic_gradient_descent(X, y, theta, learning_rate, n_epocs):
    m = len(y)
    for epoch in range(n_epocs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T @ (xi @ theta - yi)
            theta = theta - learning_rate * gradients
    return theta

# initialize parameters
theta = np.random.randn(2, 1)
learning_rate = 0.01
n_epocs = 50

# Train the model using Stochastic Gradient Descent
theta_optimal = stochastic_gradient_descent(X_b, y, theta, learning_rate, n_epocs)
print("Optimal parameters (theta):", theta_optimal)