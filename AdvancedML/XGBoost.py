import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load Dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
#Convert data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)




param = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,
    'eta': 0.1
    
}   
xgb_model = xgb.train(param, dtrain, num_boost_round=100)
y_pred = (xgb_model.predict(dtest) > 0.5).astype(int)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Classifier Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

#Define hyperparameter grid for GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'eta': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'colsample_bytree': [0.8, 1.0],
}

# Initialize XGBoost Classifier
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)

# Perform Grid SearchCV
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
#Display best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

print("Best Cross-Validation Accuracy:", grid_search.best_score_)
#Use the best hyperparameters to train a new model
best_xgb_classifier = xgb.XGBClassifier(**grid_search.best_params_, objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
best_xgb_classifier.fit(X_train, y_train)
#Make predictions with the best model
y_pred_best = best_xgb_classifier.predict(X_test)
#Evaluate the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Best XGBoost Classifier Accuracy: {accuracy_best:.2f}")
print("Classification Report for Best Model:\n", classification_report(y_test, y_pred_best))
