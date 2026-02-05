import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the titanic dataset
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)
df=pd.read_csv(url)
#  inspect Data
#print(df.head())
#print(df.info())
print(df.describe())
#hundle missing values
df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

#Remove duplicates
df.drop_duplicates()

# Filter data : Passengers in first class
first_class_passengers = df[df["Pclass"] == 1]
print("first_class_passengers \n", first_class_passengers)

#Bar Chart : Survival count by class
survival_by_class= df.groupby("Pclass")["Survived"].mean()
survival_by_class.plot(kind="bar")
plt.xlabel("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Passenger Class")
plt.show()

#Histogram : Age distribution
sns.histplot(df["Age"], bins=20, kde=True, color="blue")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution of Passengers")
plt.show()

#Scatter Plot : Age vs Fare
plt.scatter(df["Age"],df["Fare"], alpha=0.5, color="green")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Age vs Fare by Survival Status")
plt.show()

