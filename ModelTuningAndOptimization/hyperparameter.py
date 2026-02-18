from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from  sklearn.ensemble import RandomForestClassifier

#load dataset
iris = load_breast_cancer()
X, y = iris.data, iris.target

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
#Display dataset information
print("Dataset Shape:", X.shape)
print("Feature Names:", iris.feature_names)

#Train a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
#Make predictions
y_pred = rf_model.predict(X_test)
#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Classifier Accuracy: {accuracy:.8f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

#Train a Random Forest Classifier with GridSearchCV
rf_tuned = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    random_state=42
)

rf_tuned.fit(X_train, y_train)


rf_pred_tuned = rf_tuned.predict(X_test,)
accuracy_tuned = accuracy_score(y_test, rf_pred_tuned)
print(f"Tuned Random Forest Classifier Accuracy: {accuracy_tuned:.8f}")
print("Classification Report for Tuned Model:\n", classification_report(y_test, rf_pred_tuned))