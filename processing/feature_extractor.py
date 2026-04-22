import numpy as np

def extract_features(landmarks):
    if landmarks is None:
        return None

    base_x, base_y, _ = landmarks[0]

    features = []
    for x, y, z in landmarks:
        features.extend([
            x - base_x,
            y - base_y
        ])

    return np.array(features)  # shape: (42,)