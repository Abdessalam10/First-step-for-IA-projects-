from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Load the Iris dataset
iris = load_iris()
# Create a DataFrame from the Iris dataset
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target

#Display dataset information
print("Dataset Information:")
print(X.describe())
print("\n Target Classes", iris.target_names)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train K-NN Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
# Predict on test data
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of K-NN model without Scaling: {accuracy:.2f}")
print("Classification Report without Scaling:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

#apply Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled data into training and testing sets
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Train K-NN Model on scaled data
knn_model_scaled = KNeighborsClassifier(n_neighbors=5)
knn_model_scaled.fit(X_train_scaled, y_train)
# Predict on scaled test data
y_pred_scaled = knn_model_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
print(f"Accuracy of K-NN model with Min-Max Scaling: {accuracy_scaled:.2f}")
print("Classification Report with Min-Max Scaling:\n", classification_report(y_test, y_pred_scaled, target_names=iris.target_names))

#apply Standard Scaling
scaler = StandardScaler()
X_scaled_standard = scaler.fit_transform(X)
# Split the standard scaled data into training and testing sets 
X_train_standard, X_test_standard, y_train, y_test = train_test_split(X_scaled_standard, y, test_size=0.2, random_state=42)
#Train K-NN Model on standard scaled data
knn_model_standard = KNeighborsClassifier(n_neighbors=5)    
knn_model_standard.fit(X_train_standard, y_train)
# Predict on standard scaled test data
y_pred_standard = knn_model_standard.predict(X_test_standard)
accuracy_standard = accuracy_score(y_test, y_pred_standard)
print(f"Accuracy of K-NN model with Standard Scaling: {accuracy_standard:.2f}")
print("Classification Report with Standard Scaling:\n", classification_report(y_test, y_pred_standard, target_names=iris.target_names))