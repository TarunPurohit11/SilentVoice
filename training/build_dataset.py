import os
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data", "asl_dataset")
)

print("Resolved DATA_DIR:", DATA_DIR)

X, y = [], []

detected = 0
skipped = 0

# ----------------------------
# ANGLE FUNCTION
# ----------------------------
def get_angle(a, b, c):
    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )

    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


# ----------------------------
# PROCESS DATASET
# ----------------------------
for label in os.listdir(DATA_DIR):

    folder_path = os.path.join(DATA_DIR, label)

    if not os.path.isdir(folder_path):
        continue

    print(f"\nProcessing label: {label}")

    for img_name in os.listdir(folder_path):

        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(folder_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            skipped += 1
            continue

        image = cv2.resize(image, (512, 512))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            skipped += 1
            continue

        detected += 1

        # ----------------------------
        # LANDMARKS
        # ----------------------------
        lm = result.multi_hand_landmarks[0].landmark
        landmarks = np.array([[p.x, p.y, p.z] for p in lm])

        # ----------------------------
        # NORMALIZATION
        # ----------------------------
        base = landmarks[0]
        landmarks = landmarks - base

        max_val = np.max(np.abs(landmarks))
        if max_val > 0:
            landmarks = landmarks / max_val

        # ----------------------------
        # XY FEATURES (42)
        # ----------------------------
        xy_features = landmarks[:, :2].flatten()

        # ----------------------------
        # ANGLE FEATURES (~12)
        # ----------------------------
        joints = [
            (0, 5, 6), (5, 6, 7), (6, 7, 8),
            (0, 9,10), (9,10,11), (10,11,12),
            (0,13,14), (13,14,15), (14,15,16),
            (0,17,18), (17,18,19), (18,19,20),
        ]

        angles = []
        for a, b, c in joints:
            angle = get_angle(landmarks[a], landmarks[b], landmarks[c])
            angles.append(angle)

        # ----------------------------
        # FINAL FEATURES (54)
        # ----------------------------
        features = np.concatenate([xy_features, angles])

        X.append(features)
        y.append(label.upper())

# ----------------------------
# SAVE DATA
# ----------------------------
X = np.array(X)
y = np.array(y)

print("\nFINAL DATASET SIZE:", X.shape)
print("Detected:", detected)
print("Skipped:", skipped)

np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("Dataset saved successfully ✅")