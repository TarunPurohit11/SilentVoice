import joblib

class StaticModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.encoder = joblib.load("models/label_encoder.pkl")

    def predict(self, features):
        pred = self.model.predict([features])
        return self.encoder.inverse_transform(pred)[0]