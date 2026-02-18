from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
# Load the iris dataset
iris = load_iris()
X,y = iris.data, iris.target

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train individual models
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train_scaled, y_train)
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train_scaled, y_train)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)
#Make predictions
logistic_pred = logistic_model.predict(X_test_scaled)
tree_pred = tree_model.predict(X_test_scaled)
knn_pred = knn_model.predict(X_test_scaled)


#creating a Voting Classifier
voting_clf = VotingClassifier(estimators=[('lr', logistic_model), ('dt', tree_model), ('knn', knn_model)], voting='hard')
voting_clf.fit(X_train_scaled, y_train)

#Predict with ensemble model
voting_pred = voting_clf.predict(X_test_scaled)
print("Voting Classifier Predictions:\n", voting_pred)

#Evaluate the ensemble model
accuracy = accuracy_score(y_test, voting_pred)
print(f"Ensemble Model Accuracy: {accuracy:.2f}")

#Evaluate individual models
logistic_accuracy = accuracy_score(y_test, logistic_pred)
tree_accuracy = accuracy_score(y_test, tree_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)

print(f"Logistic Regression Accuracy: {logistic_accuracy:.2f}")
print(f"Decision Tree Accuracy: {tree_accuracy:.2f}")
print(f"KNN Accuracy: {knn_accuracy:.2f}")