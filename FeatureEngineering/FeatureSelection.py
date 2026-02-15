from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection  import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#Load the diabetes dataset
diabetes = load_diabetes()
df=pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

#Display dataset information
print(df.info())
#Display summary statistics
print(df.describe())

#Calculate correlation matrix
correlation_matrix = df.corr()

#Plot heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=    0.5)
plt.title('Correlation Matrix Heatmap') 
plt.show()

#Select features with correlation above a certain threshold
correlation_features = correlation_matrix['target'].sort_values(ascending=False)
print("Correlation of features with target variable:\n", correlation_features)  


#Seperate features and target variable
X = df.drop(columns=['target'])
y = df['target']

#Calculate mutual information scores
mutual_info_scores = mutual_info_regression(X, y)
#Create a DataFrame for better visualization of mutual information 
mutual_info_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information ': mutual_info_scores})
#Sort the DataFrame by mutual information scores
mutual_info_df = mutual_info_df.sort_values(by='Mutual Information ', ascending=False)
print("Mutual Information Scores:\n", mutual_info_df)

#Train a Random Forest Regressor to evaluate feature importance
model = RandomForestRegressor( random_state=42)
model.fit(X, y)

#Get feature importance scores
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importance from Random Forest:\n", importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.title('Feature Importance from Random Forest')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
plt.show()