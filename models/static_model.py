import joblib
import numpy as np

class StaticModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.encoder = joblib.load("models/label_encoder.pkl")

    def predict(self, features):
        if features is None:
            return "", 0

        proba = self.model.predict_proba([features])[0]
        max_prob = max(proba)

        pred_index = proba.argmax()
        label = self.encoder.inverse_transform([pred_index])[0]

        return label, max_prob