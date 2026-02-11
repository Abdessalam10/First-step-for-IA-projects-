#EDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
data = pd.read_csv(url)
#regression linaire
## Define variables
X = data[["total_bill"]].values.reshape(-1, 1)  # Reshape for sklearn
y = data["tip"].values
## Fir linear Regression
model = LinearRegression()
model.fit(X, y)
#output coefficients
print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"R-squared: {model.score(X, y)}")
#Plot the regression line
sns.scatterplot(x="total_bill", y="tip", data=data, color="blue", label="Data Points")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.legend()
plt.show()
del data["smoker"]
del data["day"]
del data["time"]
#Conducting Hypothesis Testing

#Seperate data by gender
male_data = data[data["sex"] == "Male"]["tip"]
female_data = data[data["sex"] == "Female"]["tip"]

#Perform t-test
t_statistic, p_value = ttest_ind(male_data, female_data)
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference in tips between genders.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in tips between genders.")



del data["sex"]

# Visualize Distribution
sns.histplot(data["total_bill"], kde=True, color="blue")
plt.title("Distribution of Total Bill")
plt.xlabel("Total Bill")
plt.ylabel("Frequency")
plt.show()

#Correlation heatmap
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()




