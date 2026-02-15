import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#Load Titanic dataset
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_data = pd.read_csv(url)
df = pd.DataFrame(titanic_data)
#hundle missing values
df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])


#Preview the first few rows of the dataset
print(df.head())

"One-Hot Encoding"
# Perform one-hot encoding on the 'Sex' column
df_encoded_hot = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print(df_encoded_hot.head())

#apply label encoding 
label_encoder = LabelEncoder()
df['Pclass_encoded'] = label_encoder.fit_transform(df['Pclass'])
print(df[['Pclass', 'Pclass_encoded']].head())

#Apply Frequency Encoding
df['Ticket_freq'] = df['Ticket'].map(df['Ticket'].value_counts())
print(df[['Ticket', 'Ticket_freq']].head())

X = df_encoded_hot.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df_encoded_hot['Survived']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
#Make predictions
y_pred = model.predict(X_test)
#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")