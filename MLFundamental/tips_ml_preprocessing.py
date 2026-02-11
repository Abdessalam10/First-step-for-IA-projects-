import sympy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"

data= pd.read_csv(url)

featuers = data[['total_bill', 'tip', 'size']]
target = data['tip']

print(featuers.head())
print(target.head())

X_train, X_test, y_train, y_test = train_test_split(featuers, target, test_size=0.2, random_state=42)

print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)

# Visualize relations
sns.pairplot(data, x_vars=['total_bill', 'size'], y_vars='tip', height=5, aspect=0.8, kind="scatter")
plt.title("Features vs Target")
plt.show()