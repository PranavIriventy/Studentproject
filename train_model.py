import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Create total score
df["total score"] = df["math score"] + df["reading score"] + df["writing score"]

# Features and target
X = df[["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course",
        "math score", "reading score", "writing score"]]
y = df["total score"]

# One-hot encode categorical features
X_encoded = pd.get_dummies(X)

# Save column names for later in app
joblib.dump(X_encoded.columns, "model_columns.pkl")

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_encoded, y)

# Save model
joblib.dump(model, "student_performance_model.pkl")

print("✅ Model and columns saved!")
