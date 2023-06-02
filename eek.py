import cv2
import numpy as np
import time

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')


# Function to detect faces and hands
def detect_faces_hands(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Detect hands
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Draw rectangles around the detected hands
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, len(faces) > 0, len(hands) > 0


# Open webcam
cap = cv2.VideoCapture(0)

face_start_time = None
face_duration = 0

gaze_start_time = None
gaze_duration = 0
looking_at_camera = False

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Break the loop if no frame is captured
    if not ret:
        break

    # Detect faces and hands in the frame
    output_frame, face_detected, hand_detected = detect_faces_hands(frame)

    # Update face duration
    if face_detected:
        if face_start_time is None:
            face_start_time = time.time()
        face_duration = time.time() - face_start_time

        # Check if face is looking at the camera
        if not looking_at_camera:
            gaze_start_time = time.time()
            looking_at_camera = True
    else:
        face_start_time = None
        face_duration = 0

        # Update gaze duration
        if looking_at_camera:
            gaze_duration = time.time() - gaze_start_time
            looking_at_camera = False

    # Update hand duration
    if hand_detected:
        if hand_start_time is None:
            hand_start_time = time.time()
        hand_duration = time.time() - hand_start_time
    else:
        hand_start_time = None
        hand_duration = 0

    # Display the output frame
    cv2.imshow('Face and Hand Detection', output_frame)

    # Log the duration of face, hands, and gaze
    print("Face duration: {:.2f}s".format(face_duration))
    print("Hand duration: {:.2f}s".format(hand_duration))
    print("Gaze duration: {:.2f}s".format(gaze_duration))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
