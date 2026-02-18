from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#Load Dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train a Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=42)
gb_classifier.fit(X_train, y_train)
#Make predictions   
y_pred = gb_classifier.predict(X_test)
#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Gradient Boosting Classifier Accuracy: {accuracy:.2f}")

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    

}
grid_search = GridSearchCV(estimator=gb_classifier,
                           param_grid=param_grid,   
                            cv=5,
                            scoring='accuracy',
                            n_jobs=-1,
                            )
grid_search.fit(X_train, y_train)
#Display best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", f"{grid_search.best_score_}")


#Use the best hyperparameters to train a new model
best_gb_classifier = GradientBoostingClassifier(**grid_search.best_params_, random_state=42)
best_gb_classifier.fit(X_train, y_train)
#Make predictions with the best model
y_pred_best = best_gb_classifier.predict(X_test)
#Evaluate the best model
accuracy_best = accuracy_score(y_test, y_pred_best)

