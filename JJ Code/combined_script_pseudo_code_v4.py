# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 20:51:24 2024

@author: 20Jan
"""

import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist

# Load the TensorFlow model
model = load_model('/Users/20Jan/Junior Jay Capstone/JJ Code/model_optimal2.h5')

# Initialize Dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Mouth Aspect Ratio (MAR) calculation
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[14], mouth[18])
    B = dist.euclidean(mouth[12], mouth[16])
    C = dist.euclidean(mouth[10], mouth[20])
    return (A + B) / (2.0 * C)

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for rect in rects:
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        
        # Get facial landmarks
        shape = predictor(gray, rect)
        shape_np = np.array([[p.x, p.y] for p in shape.parts()])

        # Compute EAR and MAR
        leftEye = shape_np[42:48]
        rightEye = shape_np[36:42]
        mouth = shape_np[48:68]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Emotion detection
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype("float") / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        emotion_prediction = model.predict(face_img)
        emotion = np.argmax(emotion_prediction)

        # Display results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
