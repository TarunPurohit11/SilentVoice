import numpy as np

def extract_features(landmarks):
    if landmarks is None:
        return None

    landmarks = np.array(landmarks)

    # ----------------------------
    # Normalize (same as before)
    # ----------------------------
    base = landmarks[0]
    landmarks = landmarks - base

    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks = landmarks / max_val

    # ----------------------------
    # XY features
    # ----------------------------
    xy_features = landmarks[:, :2].flatten()

    # ----------------------------
    # ANGLE FEATURES (NEW 🔥)
    # ----------------------------
    angles = []

    def get_angle(a, b, c):
        ba = a - b
        bc = c - b

        cos_angle = np.dot(ba, bc) / (
            np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
        )

        return np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # finger joints (MediaPipe indices)
    joints = [
        (0, 5, 6), (5, 6, 7), (6, 7, 8),      # index
        (0, 9,10), (9,10,11), (10,11,12),     # middle
        (0,13,14), (13,14,15), (14,15,16),    # ring
        (0,17,18), (17,18,19), (18,19,20),    # pinky
    ]

    for a, b, c in joints:
        angle = get_angle(landmarks[a], landmarks[b], landmarks[c])
        angles.append(angle)

    # ----------------------------
    # FINAL FEATURES
    # ----------------------------
    return np.concatenate([xy_features, angles])