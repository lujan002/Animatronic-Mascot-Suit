# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 20:58:00 2024

@author: 20Jan
"""

import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model
from imutils import face_utils

# Load models
model = load_model('/Users/20Jan/Junior Jay Capstone/JJ Code/model_optimal2.h5')
predictor = dlib.shape_predictor('/Users/20Jan/Junior Jay Capstone/JJ Code/shape_predictor_68_face_landmarks.dat')
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Initialize the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Functions to calculate the Eye Aspect Ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize the region of interest for emotion recognition
        roi = cv2.resize(face_roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # Emotion detection
        predictions = model.predict(roi)
        emotion = emotion_dict[np.argmax(predictions)]
        
        # Put the emotion label above the rectangle
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Detecting blinks with Dlib landmarks
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[42:48]
        rightEye = shape[36:42]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        # Draw eye contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Display EAR
        cv2.putText(frame, f'EAR: {ear:.2f}', (x + 100, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face and Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
