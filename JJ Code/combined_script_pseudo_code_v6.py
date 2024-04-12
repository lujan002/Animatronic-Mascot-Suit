# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:00:16 2024

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

# Constants for blink and mouth open detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2
MOUTH_AR_THRESH = 0.79

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# Initialize emotion stability control
emotion_stability_buffer = []
stable_emotion = ["Neutral"]
stability_frames = 2  # Number of frames to keep emotion stable

# Bias factor adjustments
SAD_INDEX = 5  # Assuming 'Sad' is at index 5
SAD_BIAS_FACTOR = 2  # Increase probability by 50%

# Function to adjust predictions
def adjust_predictions(predictions):
    predictions[0][SAD_INDEX] *= SAD_BIAS_FACTOR
    return predictions

# Functions to calculate the Eye Aspect Ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Mouth Aspect Ratio (MAR) calculation
def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar

def remove_classes(predictions, classes_to_remove):
    predictions[:, classes_to_remove] = 0
    sum_predictions = np.sum(predictions, axis=1, keepdims=True)
    return predictions / sum_predictions if np.any(sum_predictions) else predictions

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Identify the most central face
    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: (x[2] * x[3]), reverse=True)  # Sort faces based on area (w*h)
        x, y, w, h = faces[0]  # Consider only the largest face
    # for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize the region of interest for emotion recognition
        roi = cv2.resize(face_roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # Emotion detection
        predictions = model.predict(roi)
        predictions = remove_classes(predictions, [1,2]) # remove 'disgust' and 'fear' labels
        predictions = adjust_predictions(predictions)
        emotion = emotion_dict[np.argmax(predictions)]
        
        # Update emotion stability buffer
        if len(emotion_stability_buffer) == 0 or emotion == emotion_stability_buffer[-1]:
            emotion_stability_buffer.append(emotion)
        else:
            emotion_stability_buffer = [emotion]  # Reset the buffer with the new emotion
       
        # Determine stable emotion
        if len(emotion_stability_buffer) >= stability_frames:
            stable_emotion.append(emotion_stability_buffer[0])  # The stable emotion
            emotion_stability_buffer = []  # Reset buffer after updating stable emotion
        # else:
        #     stable_emotion = emotion_stability_buffer[-1] if emotion_stability_buffer else emotion  # Use the most recent until stable

        # Put the emotion label above the rectangle
        cv2.putText(frame, stable_emotion[-1], (x+20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        # Show prediction values
        print(predictions)
        # predictions = predictions.flatten()
        # predictions_text = ' '.join([f"{pred:.2f}" for pred in predictions])
        # cv2.putText(frame, predictions, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Detecting blinks with Dlib landmarks
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Calculate eye aspect ratio
        leftEye = shape[42:48]
        rightEye = shape[36:42]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        # Calculate mouth aspect ratio
        mouth = shape[49:68]
        mar = mouth_aspect_ratio(mouth)
        
        # Draw eye contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Draw mouth contours
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        
        # Display EAR, MAR on frame
        cv2.putText(frame, f'EAR: {ear:.2f}', (x+w, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (x+w, y+70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            # otherwise, the eye aspect ratio is not below the blink
        # threshold
            cv2.putText(frame, "Eyes are Closed!", (x+w,y+40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            # reset the eye frame counter
            COUNTER = 0
            
        # Draw text if mouth is open
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Mouth is Open!", (x+w,y+100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            
    # Display the resulting frame
    cv2.imshow('Face and Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
