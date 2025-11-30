import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1) Load data
df = pd.read_csv("Data/listeningData_encoded.csv")

# 2) Features/target (exclude other mental disorders)
X = df.drop(columns=["Anxiety", "Depression", "Insomnia", "OCD"])
y = df["Anxiety"]

# 3) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Model
model = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    verbose=100
)

# 5) Fit
model.fit(X_train, y_train)

# 6) Predict
y_pred = model.predict(X_test)

# 7) Evaluate (manual RMSE to avoid version issues)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)

# 8) Feature importance
importances = model.get_feature_importance()
for feat, imp in zip(X.columns, importances):
    print(f"{feat}: {imp:.4f}")