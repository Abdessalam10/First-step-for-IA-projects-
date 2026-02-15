import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#Load Bike Sharing dataset
url = "day.csv"
df = pd.read_csv(url)

print(df.head())

# Convert dteday to datetime
df['dteday'] = pd.to_datetime(df['dteday'])

#create new features
df['date_of_week'] = df['dteday'].dt.day_name()
df['month'] = df['dteday'].dt.month
df['year'] = df['dteday'].dt.year

print(df[['dteday', 'date_of_week', 'month', 'year']].head())


# Select features and target variable
X = df[['temp']]
y = df['cnt']

#Apply polynomial feature transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
#Display the transformed features
print("Original and polynomial features:\n")

print(pd.DataFrame(X_poly, columns=['temp','temp^2']).head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

X_poly_train, X_poly_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)    
model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train)
y_poly_pred = model_poly.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_poly_pred)
r2_poly = r2_score(y_test, y_poly_pred)
print(f"Polynomial Mean Squared Error: {mse_poly:.2f}")
print(f"Polynomial R-squared: {r2_poly:.2f}")