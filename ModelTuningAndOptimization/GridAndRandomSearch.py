from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV   
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Display dataset information
print("Dataset Shape:", X.shape)
print("Feature Names:", iris.feature_names)
#Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Define hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
 
}
#Initialize GirdSearchCV with Random Forest Classifier
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid, 
                           cv=5, 
                           scoring='accuracy',
                           n_jobs=-1)
#Perform Grid SearchCV
grid_search.fit(X_train, y_train)
#Display best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)
print("Best Estimator:", grid_search.best_estimator_)

#Define hyperparameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': np.arange(50,200,100),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 15]
}
#Initialize RandomizedSearchCV with Random Forest Classifier
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),   
                                      param_distributions=param_dist, 
                                      n_iter=10, 
                                      cv=5, 
                                      scoring='accuracy',
                                      n_jobs=-1,
                                      random_state=42)
#Perform Randomized SearchCV
random_search.fit(X_train, y_train)
#Display best hyperparameters
print("Best Hyperparameters from RandomizedSearchCV:", random_search.best_params_)
print("Best Cross-Validation Accuracy from RandomizedSearchCV:", random_search.best_score_)
print("Best Estimator from RandomizedSearchCV:", random_search.best_estimator_)
