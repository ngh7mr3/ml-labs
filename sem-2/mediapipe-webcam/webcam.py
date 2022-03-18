#!python3
import cv2
import mediapipe as mp
import numpy as np

def webcam_stream():
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            # cv2.imshow('Input', frame)
            # print('type of frame', type(frame))
            yield frame
    except Exception:
        pass

    cap.release()
    cv2.destroyAllWindows()

def is_ok(hand_landmarks) -> bool:
    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    p_finger =  hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    a, b = np.array([thumb.x, thumb.y, thumb.z]), np.array([p_finger.x, p_finger.y, p_finger.z])

    dst = np.linalg.norm(a-b)
    print('calculated dst', dst)
    confidence = 0.05
    
    return dst < confidence


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.7) as hands:
    for frame in webcam_stream():
        results = hands.process(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1))

        oks = []
        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            annotated = cv2.flip(frame.copy(), 1)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                if is_ok(hand_landmarks):
                    oks.append('OK')
        else:
            annotated = cv2.flip(frame.copy(), 1)

        #annotated = cv2.resize(annotated, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
        if oks:
            annotated = cv2.putText(annotated, ' '.join(oks), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Hands', annotated)
        cv2.waitKey(1)

