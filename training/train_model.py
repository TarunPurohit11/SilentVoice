import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Create models folder
# ----------------------------
os.makedirs("models", exist_ok=True)

# ----------------------------
# Load dataset
# ----------------------------
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# ----------------------------
# Encode labels
# ----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ----------------------------
# Train model
# ----------------------------
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# ----------------------------
# Save model + encoder
# ----------------------------
joblib.dump(model, "models/alphabet_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("Model and encoder saved")