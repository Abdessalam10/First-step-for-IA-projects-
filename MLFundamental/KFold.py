from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
# Load the Iris dataset
data= load_iris()
X, y=data.data, data.target

# Initialize the Random Forest Classifier
model= RandomForestClassifier(n_estimators=100, random_state=42)
# Set up K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())