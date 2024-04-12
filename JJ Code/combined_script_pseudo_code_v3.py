# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:58:53 2024

@author: 20Jan
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
from tensorflow.keras.models import load_model
import time

# Load your custom model
model = load_model('/Users/20Jan/Junior Jay Capstone/JJ Code/model_optimal2.h5')
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load Dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55
    return (A + B) / (2.0 * C)

def remove_classes(predictions, classes_to_remove):
    predictions[:, classes_to_remove] = 0
    sum_predictions = np.sum(predictions, axis=1, keepdims=True)
    return predictions / sum_predictions if np.any(sum_predictions) else predictions

# Constants for eye and mouth detection
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79

# Video stream setup
cap = cv2.VideoCapture(0)
time.sleep(1.0)  # Give some time for the camera to warm-up

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    rects = detector(gray, 0)
    for rect in rects:
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        
        # Determine the facial landmarks
        shape = predictor(gray, rect)
        shape_np = face_utils.shape_to_np(shape)
        
        # Eye and mouth aspects
        leftEye = shape_np[face_utils.FACIAL_LANDMARKS_IDXS['left_eye'][0]:face_utils.FACIAL_LANDMARKS_IDXS['left_eye'][1]]
        rightEye = shape_np[face_utils.FACIAL_LANDMARKS_IDXS['right_eye'][0]:face_utils.FACIAL_LANDMARKS_IDXS['right_eye'][1]]
        mouth = shape_np[face_utils.FACIAL_LANDMARKS_IDXS['mouth'][0]:face_utils.FACIAL_LANDMARKS_IDXS['mouth'][1]]
        
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        mar = mouth_aspect_ratio(mouth)
        
        # Draw on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {mar:.2f}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Emotion detection
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype("float") / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        
        predictions = model.predict(face_img)
        predictions = remove_classes(predictions, [1, 2])  # Assuming 'disgust' and 'fear' need removal
        emotion = emotion_dict[np.argmax(predictions)]
        cv2.putText(frame, emotion, (x, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Emotion and Face Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
