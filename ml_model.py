# ml_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
df = pd.read_csv("your_data.csv")  # <-- replace with your actual file

# Example: Drop non-numeric or irrelevant columns
df = df.select_dtypes(include='number').dropna()

# Split features and target
X = df.drop("Mental_Health_Score", axis=1)
y = df["Mental_Health_Score"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, "model.joblib")
