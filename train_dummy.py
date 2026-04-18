import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("models", exist_ok=True)

X = np.random.rand(100, 63)
y = np.random.choice(['A', 'B', 'C'], 100)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'models/alphabet_model.pkl')

print("Dummy model created!")