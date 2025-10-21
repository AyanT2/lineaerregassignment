import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

RANDOM_SEED = 42

print("--- Loading Data ---")
galton_data = sm.datasets.get_rdataset("GaltonFamilies", "HistData")
df = galton_data.data

print("\n--- Initial Data Inspection ---")
print("First 5 records:")
print(df.head())
print("\nDataset Info:")
df.info()

print("\n--- Checking for Missing Values ---")
print(df.isnull().sum())

df = pd.get_dummies(df, columns=['gender'], drop_first=True)

# Define features (X) and target (y)
features = ['father', 'mother', 'gender_male']
target = 'childHeight'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
print(f"\n--- Data Split ---")
print(f"Training set size: {len(X_train)} records")
print(f"Test set size: {len(X_test)} records")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

print("\n--- Training Models ---")

ols_model = LinearRegression()
ols_model.fit(X_train_scaled, y_train)
print("OLS Baseline Model Trained.")

ridge_params = {'alpha': np.logspace(-4, 4, 50)}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train_scaled, y_train)
best_ridge_model = ridge_grid.best_estimator_
print(f"Ridge Model Trained. Best alpha: {ridge_grid.best_params_['alpha']:.4f}")

lasso_params = {'alpha': np.logspace(-4, 4, 50)}
lasso_grid = GridSearchCV(Lasso(), lasso_params, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train_scaled, y_train)
best_lasso_model = lasso_grid.best_estimator_
print(f"Lasso Model Trained. Best alpha: {lasso_grid.best_params_['alpha']:.4f}")

print("\n--- Evaluating Models on the Test Set ---")

models = {
    "OLS": ols_model,
    "Ridge": best_ridge_model,
    "Lasso": best_lasso_model
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R-squared": r2
    })

results_df = pd.DataFrame(results)
print("\n--- Final Test Scores ---")
print(results_df.round(4))

print("\n--- Generating Visualizations ---")

y_pred_ols = ols_model.predict(X_test_scaled)
residuals = y_test - y_pred_ols

plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_ols, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs. Fitted Values', fontsize=16)
plt.xlabel('Fitted Values (Predicted Child Height)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_ols)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual vs. Predicted Height', fontsize=16)
plt.xlabel('Actual Child Height (inches)', fontsize=12)
plt.ylabel('Predicted Child Height (inches)', fontsize=12)
plt.axis('equal')
plt.show()

coefficients = pd.DataFrame({
    'Feature': features,
    'OLS': ols_model.coef_,
    'Ridge': best_ridge_model.coef_,
    'Lasso': best_lasso_model.coef_
})

coefficients_melted = coefficients.melt(id_vars='Feature', var_name='Model', value_name='Coefficient')

plt.figure(figsize=(12, 7))
sns.barplot(x='Feature', y='Coefficient', hue='Model', data=coefficients_melted, palette='viridis')
plt.title('Standardized Coefficients by Model', fontsize=16)
plt.ylabel('Coefficient Value', fontsize=12)
plt.xlabel('Feature', fontsize=12)
plt.axhline(0, color='black', linewidth=0.8)
plt.show()

print("\n--- Script Finished ---")