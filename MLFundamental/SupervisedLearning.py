# Task 1 : Perform and Preprocess the data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the California housing dataset
california = fetch_california_housing(as_frame=True)    
df = california.frame
#define features and target variable
X= df [['MedInc', 'HouseAge', 'AveRooms']]
y= df['MedHouseVal']
#Inspect data
print(df.info())
print(df.describe())

# visualize relationship between features and target variable
#sns.pairplot(df, vars = ['MedInc', 'HouseAge', 'AveRooms','MedHouseVal'])
#plt.show()

#Check for missing values
print("Missing values in each column:")

# Load Telco Customer Churn dataset
df_telco = pd.read_csv('telco.csv')

# Select only the required columns
selected_columns = ['customerID', 'gender', 'tenure', 'MonthlyCharges', 
                   'TotalCharges', 'Contract', 'PaymentMethod', 'Churn']
df_telco = df_telco[selected_columns]
# Encode categorical variables
label_encoder = LabelEncoder()
df_telco['Churn'] = label_encoder.fit_transform(df_telco['Churn'])
print(df_telco.info())
# Define features and target variable
X1 = df_telco.drop(['Churn'], axis=1)
y1 = df_telco['Churn']

# Convert TotalCharges to numeric (it's stored as string)
X1['TotalCharges'] = pd.to_numeric(X1['TotalCharges'], errors='coerce')

# Force encode all string columns to numerical
# Explicitly encode each string column
string_columns = ['customerID', 'gender', 'Contract', 'PaymentMethod']
for col in string_columns:
    if col in X1.columns:
        le = LabelEncoder()
        X1[col] = le.fit_transform(X1[col].astype(str))

# Double-check: encode any remaining object columns
for col in X1.columns:
    if X1[col].dtype == 'object':
        le = LabelEncoder()
        X1[col] = le.fit_transform(X1[col].astype(str))

print("Final data types:")
print(X1.dtypes)
print("Final sample data:")
print(X1.head())

# Fill missing values only for numeric columns
numeric_columns = X1.select_dtypes(include=['float64', 'int64']).columns
X1[numeric_columns] = X1[numeric_columns].fillna(X1[numeric_columns].mean())

# Scale Features
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)
# Split Dataset
X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1, test_size=0.2, random_state=42)
# Train Logistic Regression Model
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X1_train, y1_train)
# Train K-NN Model
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X1_train, y1_train)

# Make Predictions
y1_pred = logistic_model.predict(X1_test)
y1_pred_knn = knn_model.predict(X1_test)

print("Logistic Regression Performance:")
print(classification_report(y1_test, y1_pred))
print("Confusion Matrix:")
print(confusion_matrix(y1_test, y1_pred))
print("K-NN Performance:")
print(classification_report(y1_test, y1_pred_knn))
print("Confusion Matrix:")
print(confusion_matrix(y1_test, y1_pred_knn))
# Evaluate Performance
print("Classification Report:")
print(classification_report(y1_test, y1_pred))
print("Confusion Matrix:")
print(confusion_matrix(y1_test, y1_pred))
# Inspect data
print(df_telco.info())
print(df_telco.describe())

# isualize churn distribution
sns.countplot(x='Churn', data=df_telco)
plt.title('Distribution of Churn')
plt.show()

# Handle missing values only for numeric columns in df_telco
numeric_cols = df_telco.select_dtypes(include=['float64', 'int64']).columns
df_telco[numeric_cols] = df_telco[numeric_cols].fillna(df_telco[numeric_cols].mean())

#Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

#Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
#Make Prediction
y_pred = model.predict(X_test)
#Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

