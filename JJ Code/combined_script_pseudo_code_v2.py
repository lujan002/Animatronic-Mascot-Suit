# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:26:53 2024

@author: 20Jan
"""

import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist

# Load models
model = load_model('/Users/20Jan/Junior Jay Capstone/JJ Code/model_optimal2.h5')
predictor = dlib.shape_predictor("/Users/20Jan/Junior Jay Capstone/JJ Code/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# Define the emotion dictionary
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Eye and Mouth aspect ratio calculation functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
    B = dist.euclidean(mouth[4], mouth[8]) # 53, 57
    C = dist.euclidean(mouth[0], mouth[6]) # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

def remove_classes(predictions, classes_to_remove):
    predictions[:, classes_to_remove] = 0
    sum_predictions = np.sum(predictions, axis=1, keepdims=True)
    return predictions / sum_predictions if np.any(sum_predictions) else predictions

# Constants for blink and mouth open detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2
MOUTH_AR_THRESH = 0.79

# Start the video stream
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = face_utils.shape_to_np(shape)

        leftEye = shape_np[face_utils.FACIAL_LANDMARKS_IDXS['left_eye'][0]:face_utils.FACIAL_LANDMARKS_IDXS['left_eye'][1]]
        rightEye = shape_np[face_utils.FACIAL_LANDMARKS_IDXS['right_eye'][0]:face_utils.FACIAL_LANDMARKS_IDXS['right_eye'][1]]
        mouth = shape_np[face_utils.FACIAL_LANDMARKS_IDXS['mouth'][0]:face_utils.FACIAL_LANDMARKS_IDXS['mouth'][1]]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        mar = mouth_aspect_ratio(mouth)

        # Detect blink
        if ear < EYE_AR_THRESH:
            cv2.putText(frame, "Blinking", (rect.left(), rect.top() - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Detect open mouth
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Mouth Open", (rect.left(), rect.top() - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Emotion detection
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype("float") / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        
        predictions = model.predict(face_img)
        classes_to_remove = [1, 2] # Removing 'Disgust' and 'Fear'
        predictions = remove_classes(predictions, classes_to_remove)
        emotion = emotion_dict[np.argmax(predictions)]
        
        # Draw emotion text
        cv2.putText(frame, emotion, (rect.left(), rect.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw face rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
