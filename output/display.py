import cv2

def show_prediction(frame, text):
    cv2.putText(
        frame,
        f"Prediction: {text}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    return frame