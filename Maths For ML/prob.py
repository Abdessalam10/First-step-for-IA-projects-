# Problem
# A disease affests 1% of a population 
# A test for the disease is 99% accurate.
# Find the probabability of having the disease given a positive test result

def bayes_theorem(prior,sensitivity, specificity):
    # Calculate the probability of a positive test result
    p_positive = (sensitivity * prior) + ((1 - specificity) * (1 - prior))
    
    # Calculate the probability of having the disease given a positive test result
    p_disease_given_positive = (sensitivity * prior) / p_positive
    
    return p_disease_given_positive

prior=0.01  # Prevalence of the disease
sensitivity=0.95  # True positive rate
specificity=0.90  # True negative rate

posterior_probability = bayes_theorem(prior, sensitivity, specificity)
print(f"The probability of having the disease given a positive test result is: {posterior_probability:.4f}")


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson

x= np.linspace(-10, 10, 1000)
y=norm.pdf(x, loc=0, scale=1)
plt.plot(x, y, label='Normal Distribution')
plt.title('Normal Distribution')
plt.show()

# Binomial Distribution
n=10  # number of trials
p=0.5  # probability of success
x= np.arange(0, n+1)
y=binom.pmf(x, n, p)
plt.bar(x, y, label='Binomial Distribution')
plt.title('Binomial Distribution')
plt.show()

# Poisson Distribution
lambda_=3  # average rate of occurrence 
x= np.arange(0, 15)
y=poisson.pmf(x, lambda_)
plt.bar(x, y, label='Poisson Distribution')
plt.title('Poisson Distribution')
plt.show()
