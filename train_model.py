import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
df = pd.read_csv("data/test.csv")  # make sure this file exists and path is correct

# Features and target
X = df[['age', 'bp', 'creatinine']]
y = df['dialysis']

# Train-test split (optional here)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model to disk
joblib.dump(model, 'model.pkl')
print("✅ Model saved as model.pkl")
