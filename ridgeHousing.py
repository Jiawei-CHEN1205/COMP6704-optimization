# author: CHEN Jiawei
# date: 2022-11-23
# project: Housing Price Regression


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
colnames = ["CRIM", "ZN", "INDUS", "CHAS", "NOS", "RM", "AGE","DIS","RAD","TAX", "PTRATIO","B","LSTAT","MEDV"]
df = pd.read_csv('housing.csv', delim_whitespace=True, header=None, names=colnames).head(100)
print(df)
target= ["MEDV"]
features = ["CRIM", "ZN", "INDUS", "CHAS", "NOS", "RM", "AGE","DIS","RAD","TAX", "PTRATIO","B","LSTAT"]

y = df[target]
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

y_pred = linear_reg.predict(X_test)
mse_linear = mean_squared_error(y_pred, y_test)

print(mse_linear)

print(linear_reg.coef_)

# Lasso Regression
lambda_values = [0.000001, 0.0001, 0.001, 0.005, 0.01, 0.05,  0.1, 0.2, 0.3, 0.4, 0.5]

for lambda_val in lambda_values:
    lasso_reg = Lasso(lambda_val)
    lasso_reg.fit(X_train, y_train)
    y_pred = lasso_reg.predict(X_test)
    mse_lasso = mean_squared_error(y_pred, y_test)
    print(("Lasso MSE with Lambda={} is {}").format(lambda_val, mse_lasso))
    
print(lasso_reg.coef_)

# Ridge Regression
lambda_values = [0.00001, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 3, 5, 6, 7, 8, 9, 10]

for lambda_val in lambda_values:
    ridge_reg = Ridge(lambda_val)
    ridge_reg.fit(X_train, y_train)
    y_pred = ridge_reg.predict(X_test)
    mse_ridge = mean_squared_error(y_pred, y_test)
    print(("Ridge MSE with Lambda={} is {}").format(lambda_val, mse_ridge))
    
print(ridge_reg.coef_)