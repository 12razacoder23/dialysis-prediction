import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your updated CSV file
df = pd.read_csv('data/updated_data.csv')  # path to your latest CSV

# Convert categorical columns to numeric
df['family_history'] = df['family_history'].map({'Yes': 1, 'No': 0})
df['diabetes'] = df['diabetes'].map({'Yes': 1, 'No': 0})

# Features and target
X = df[['age', 'sys_bp', 'dia_bp', 'creatinine', 'family_history', 'diabetes']]
y = df['dialysis'].map({'Yes': 1, 'No': 0})  # target also as numeric

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')
print("✅ Model retrained and saved as model.pkl")
