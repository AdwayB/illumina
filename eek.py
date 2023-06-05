import cv2
import numpy as np
import time

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')


# Function to detect faces and hands
def detect_faces_hands(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 6)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Draw rectangles around the detected eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, len(faces) > 0, len(eyes) > 0


# Open webcam
cap = cv2.VideoCapture(0)
total_face = 0
total_eyes = 0
face_start_time = None
face_duration = 0
eyes_start_time = None
eyes_duration = 0
looking_at_camera = False

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Break the loop if no frame is captured
    if not ret:
        break

    # Detect faces and gaze in the frame
    output_frame, face_detected, eyes_detected = detect_faces_hands(frame)

    # Update face duration
    if face_detected:
        if face_start_time is None:
            face_start_time = time.time()
        face_duration = time.time() - face_start_time

    else:
        total_face += face_duration
        face_start_time = None
        face_duration = 0

    # Update gaze duration
    if eyes_detected:
        if eyes_start_time is None:
            eyes_start_time = time.time()
        eyes_duration = time.time() - eyes_start_time
    else:
        total_eyes += eyes_duration
        eyes_start_time = None
        eyes_duration = 0

    # Display the output frame
    cv2.imshow('Face and Eye Detection', output_frame)

    # Log the duration of face, gaze
    print("Face duration: {:.2f}s".format(face_duration))
    print("Eye duration: {:.2f}s".format(eyes_duration))

    # Break the loop if 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

print("\n\nTotal Face duration: {:.2f}s".format(total_face))
print("Total Gaze duration: {:.2f}s".format(total_eyes))
# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# Use eye tracking to improve gaze function3
# Use gesture recognition to improve hand function
