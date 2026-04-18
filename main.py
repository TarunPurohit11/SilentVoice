import cv2
from capture.camera import Camera
from processing.hand_tracking import HandTracker
from processing.feature_extractor import extract_features
from models.static_model import StaticModel
from utils.buffer import SequenceBuffer
from output.display import show_prediction
import config

def main():
    camera = Camera()
    tracker = HandTracker()
    model = StaticModel(config.MODEL_PATH)
    buffer = SequenceBuffer(config.SEQUENCE_LENGTH)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        landmarks = tracker.get_landmarks(frame)
        features = extract_features(landmarks)

        prediction = ""

        if features is not None:
            buffer.add(features)  # future use
            prediction = model.predict(features)

        frame = show_prediction(frame, prediction)
        cv2.imshow("Sign Language Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()

if __name__ == "__main__":
    main()