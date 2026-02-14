import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score ,classification_report


#Generating synthetic data
np.random.seed(42)
n_samples = 200
X = 10 * np.random.rand(n_samples, 2)
print(X)
y = (X[:, 0] *1.5 + X[:,1]>15).astype(int)
print(y)
#Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=['Age', 'Salary'])
print(df)
df['Purchased'] = y
print(df.head())
#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['Age', 'Salary']], df['Purchased'], test_size=0.2, random_state=42)   
#Fit Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)
#Make Predictions
y_pred = model.predict(X_test)
#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)  
print("Classification Report:\n", classification_rep)   
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

#Visualize the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])    
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_test['Age'], X_test['Salary'], c=y_test, cmap=plt.cm.coolwarm, label='Data Points')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()