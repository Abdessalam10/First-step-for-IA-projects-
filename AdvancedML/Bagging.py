from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
#load the breast cancer wisconsin dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#display dataset information

print("Feature names:", cancer.feature_names)
print("Target names:", cancer.target_names)

#Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
#Make predictions
y_pred = rf_classifier.predict(X_test)
#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Classifier Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

#Define hyperparameter grid

param_grid = {  
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2',None],
   
}

grid_search = GridSearchCV(estimator=rf_classifier, 
                           param_grid=param_grid, 
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1, 
                           )
grid_search.fit(X_train, y_train)

#Display best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

