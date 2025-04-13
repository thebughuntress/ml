import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv(
    "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/BostonHousing_train.csv"
)
print(df.sample(5))

# Data separation
y = df["medv"]
X = df.drop("medv", axis=1)
print(y.sample(5))
print(X.sample(5))

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100
)

# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

# Evaluate Linear Regression
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print("Linear Regression Performance:")
print(f"Train MSE: {lr_train_mse:.4f}")
print(f"Train R² : {lr_train_r2:.4f}")
print(f"Test MSE : {lr_test_mse:.4f}")
print(f"Test R²  : {lr_test_r2:.4f}")

lr_results = pd.DataFrame(
    ["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]
).transpose()
lr_results.columns = ["Model", "Training MSE", "Training R²", "Test MSE", "Test R²"]
print(lr_results)

# Random Forest Model
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)

# Predictions
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

# Evaluate Random Forest
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

print("Random Forest Performance:")
print(f"Train MSE: {rf_train_mse:.4f}")
print(f"Train R² : {rf_train_r2:.4f}")
print(f"Test MSE : {rf_test_mse:.4f}")
print(f"Test R²  : {rf_test_r2:.4f}")

rf_results = pd.DataFrame(
    ["Random Forest", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]
).transpose()
rf_results.columns = ["Model", "Training MSE", "Training R²", "Test MSE", "Test R²"]
print(rf_results)

# Model Comparison
df_models = pd.concat([rf_results, lr_results], axis=0)
df_models = df_models.reset_index(drop=True)
print(df_models)

# Data visualization

# Linear Regression
plt.figure(figsize=(5, 5))
plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)
z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), "#F8766D")
plt.title("Actual vs Predicted with Linear Regression")
plt.ylabel("Predicted Value")
plt.xlabel("Actual Value")
plt.show()

# Random Forest
plt.figure(figsize=(5, 5))
plt.scatter(x=y_train, y=y_rf_train_pred, c="#7CAE00", alpha=0.3)
z = np.polyfit(y_train, y_rf_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), "#F8766D")
plt.title("Actual vs Predicted with Random Forest")
plt.ylabel("Predicted Value")
plt.xlabel("Actual Value")
plt.show()

# Example of Predicting a Single Value
crim = 0.35809
zn = 0.0
indus = 6.2
chas = 1
nox = 0.507
rm = 6.951
age = 88.5
dis = 2.8617
rad = 8
tax = 307
ptratio = 17.4
b = 391.7
lstat = 9.71
medv = 26.7

columns = [
    "crim",
    "zn",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "tax",
    "ptratio",
    "b",
    "lstat",
]
X_new = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
X_new_df = pd.DataFrame(X_new, columns=columns)

y_pred_lr_single = lr.predict(X_new_df)
y_pred_rf_single = rf.predict(X_new_df)

print("Actual value:", medv)
print("Predicted Value:", y_pred_lr_single[0])
print("Predicted Value:", y_pred_rf_single[0])