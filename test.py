import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

image = cv2.imread("data/asl_dataset/a/hand1_a_bot_seg_1_cropped.jpeg")

if image is None:
    print("Image not loaded")
    exit()

image = cv2.resize(image, (512, 512))
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = hands.process(rgb)

if result.multi_hand_landmarks:
    print("HAND DETECTED ✅")
    
    for hand in result.multi_hand_landmarks:
        mp_draw.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Output", image)
    cv2.waitKey(0)
else:
    print("NO HAND DETECTED ❌")