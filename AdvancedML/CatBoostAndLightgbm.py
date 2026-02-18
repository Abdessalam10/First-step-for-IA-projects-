import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier as cb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#import Titanic dataset
data = pd.read_csv('titanic.csv')
#select feature and target variable
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'
#handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

#encode categorical variables
label_encoder= {}
for col in ['Sex','Embarked']:
 le = LabelEncoder()
 data[col] = le.fit_transform(data[col])
 label_encoder[col] = le
 
# Split Data
x = data[features]
y = data[target] 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
print(f"LightGBM Classifier Accuracy: {accuracy_lgb:.4f}")
cat_features = ['Pclass', 'Sex', 'Embarked']
cb_model = cb.CatBoostClassifier(cat_features=cat_features, verbose=0)
cb_model.fit(X_train, y_train)
y_pred_cb = cb_model.predict(X_test)
accuracy_cb = accuracy_score(y_test, y_pred_cb)
print(f"CatBoost Classifier Accuracy: {accuracy_cb:.4f}")

#Train Catboost without encoding categorical features
cb_model_no_encoding = cb.CatBoostClassifier(cat_features=['Sex','Embarked'], verbose=0)    
cb_model_no_encoding.fit(X_train, y_train)
y_pred_cb_no_encoding = cb_model_no_encoding.predict(X_test)
#Evaluate the model without encoding
accuracy_cb_no_encoding = accuracy_score(y_test, y_pred_cb_no_encoding) 
print(f"CatBoost Classifier Accuracy without Encoding: {accuracy_cb_no_encoding:.4f}")
