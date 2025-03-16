import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load model and gesture classes
model = load_model('model/model.h5')
gesture_classes = ['thumbs_up', 'peace_sign', 'fist', 'open_hand', 'pointing']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            if len(landmarks) == 63:
                # Predict gesture
                prediction = model.predict(np.array([landmarks]), verbose=0)
                gesture_id = np.argmax(prediction)
                confidence = prediction[0][gesture_id]
                
                # ✅ Debugging output
                print(f"Prediction: {prediction}")
                print(f"Gesture ID: {gesture_id}")
                print(f"Confidence: {confidence}")

                if confidence > 0.5:
                    gesture_name = gesture_classes[gesture_id]
                    cv2.putText(frame, f"{gesture_name} ({confidence:.2f})", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
