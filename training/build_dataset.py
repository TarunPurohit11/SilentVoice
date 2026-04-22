import os
import cv2
import numpy as np
import mediapipe as mp

# ----------------------------
# MediaPipe setup
# ----------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ----------------------------
# Dataset path (FIXED)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data", "asl_dataset")
)

print("Resolved DATA_DIR:", DATA_DIR)
print("Exists:", os.path.exists(DATA_DIR))

allowed_labels = {"a", "b", "c", "d", "e"}

X, y = [], []

detected = 0
skipped = 0

# ----------------------------
# Check dataset folder exists
# ----------------------------
if not os.path.exists(DATA_DIR):
    print("ERROR: Dataset folder not found:", DATA_DIR)
    exit()

# ----------------------------
# Loop through dataset
# ----------------------------
for label in os.listdir(DATA_DIR):

    folder_path = os.path.join(DATA_DIR, label)

    # skip non-folders
    if not os.path.isdir(folder_path):
        continue

    # only allowed labels
    if label not in allowed_labels:
        continue

    print(f"\nProcessing label: {label}")

    images = os.listdir(folder_path)
    print("Total images:", len(images))

    for img_name in images:

        img_path = os.path.join(folder_path, img_name)

        # read image
        image = cv2.imread(img_path)
        if image is None:
            print("Unreadable image:", img_path)
            skipped += 1
            continue

        # resize
        image = cv2.resize(image, (512, 512))

        # convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # run MediaPipe
        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            skipped += 1
            continue

        detected += 1

        # extract landmarks
        lm = result.multi_hand_landmarks[0].landmark
        base = lm[0]  # wrist

        features = []
        for p in lm:
            features.extend([
                p.x - base.x,
                p.y - base.y
            ])  # removed z (noise reduction)

        X.append(features)
        y.append(label)

# ----------------------------
# Final stats
# ----------------------------
print("\n====================")
print("FINAL SAMPLES:", len(X))
print("DETECTED:", detected)
print("SKIPPED:", skipped)
print("====================\n")

# ----------------------------
# Save dataset (FIXED PATH)
# ----------------------------
if len(X) > 0:

    SAVE_DIR = os.path.abspath(
        os.path.join(BASE_DIR, "..", "data")
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    np.save(os.path.join(SAVE_DIR, "X.npy"), np.array(X))
    np.save(os.path.join(SAVE_DIR, "y.npy"), np.array(y))

    print("Saved X.npy and y.npy successfully in /data folder")
else:
    print("ERROR: No valid samples collected.")