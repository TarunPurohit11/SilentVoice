import joblib

class StaticModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, features):
        return self.model.predict([features])[0]