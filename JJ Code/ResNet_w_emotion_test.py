import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
# cv2.setUseOptimized(True)

import numpy as np
import dlib
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model
from imutils import face_utils

import sys
import io

import RPi.GPIO as GPIO
import time

import tensorflow as tf

from imutils.video import VideoStream
import argparse

import pigpio

# Define whether to run the script in headless mode
headless = False

# Bias factor adjustments
SAD_INDEX = 5  # Assuming 'Sad' is at index 5
SAD_BIAS_FACTOR = 1

# Function to adjust predictions
def adjust_predictions(predictions):
    print("Before adjustment:", predictions)
    predictions[0][SAD_INDEX] *= SAD_BIAS_FACTOR
    total = np.sum(predictions)
    predictions /= total if total > 0 else 1
    print("After adjustment:", predictions)
    return predictions

def remove_classes(predictions, classes_to_remove):
    predictions[:, classes_to_remove] = 0
    sum_predictions = np.sum(predictions, axis=1, keepdims=True)
    return predictions / sum_predictions if np.any(sum_predictions) else predictions


# Path to SSD model files
model_configuration = "/home/lujan002/Repositories/Animatronic-Mascot-Suit/JJ Code/deploy.prototxt"
model_weights = "/home/lujan002/Repositories/Animatronic-Mascot-Suit/JJ Code/res10_300x300_ssd_iter_140000.caffemodel"

# Load the SSD model
net = cv2.dnn.readNetFromCaffe(model_configuration, model_weights)

# Load the TensorFlow Lite model for emotion detection
interpreter = tf.lite.Interpreter(model_path="/home/lujan002/Repositories/Animatronic-Mascot-Suit/JJ Code/model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Emotion dictionary
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load dlib model for eye and mouth detection
predictor = dlib.shape_predictor('/home/lujan002/Repositories/Animatronic-Mascot-Suit/JJ Code/shape_predictor_68_face_landmarks.dat')

# Constants for blink and mouth open detection
EYE_AR_THRESH = 0.21
EYE_AR_CONSEC_FRAMES = 3
MOUTH_AR_THRESH = 0.76

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# Functions to calculate the Eye Aspect Ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Mouth Aspect Ratio (MAR) calculation
def mouth_aspect_ratio(mouth):
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar

# Start video capture
cap = cv2.VideoCapture('/dev/video0')
if not cap.isOpened():
    print("Failed to open the camera.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0
input_size = (300, 300)  # You can change this to other sizes like (224, 224)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 5 != 0:  # Skip every other frame
        continue
    # Convert to greyscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert frame to a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    emotions_displayed = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, w, h) = box.astype("int")
    
            # Drawing the bounding box for the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract the ROI and convert it to greyscale
            face_roi = frame[y:y+h, x:x+w]
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype(np.float32) / 255.0
            face_roi = np.expand_dims(face_roi, axis=-1)
            face_roi = np.expand_dims(face_roi, axis=0)

            # Run the emotion detection model
            interpreter.set_tensor(input_details[0]['index'], face_roi)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            predictions = remove_classes(predictions, [1,2]) # remove 'disgust' and 'fear' labels
            predictions = adjust_predictions(predictions)
            predictions = predictions.flatten()
            emotion = emotion_dict[np.argmax(predictions)]

            # Display all detected emotions in the bottom left of the window
            text_start_y = frame.shape[0] - 20
            for idx, prob in enumerate(predictions):
                emotion_label = f"{emotion_dict[idx]}: {prob:.2f}"
                cv2.putText(frame, emotion_label, (10, text_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                text_start_y -= 20

            # Detecting blinks with Dlib landmarks
            # rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            rect = dlib.rectangle(0, 0, w, h)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            if not headless:
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
                cv2.putText(frame, f'EAR: {ear:.2f}', (10, text_start_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                # cv2.putText(frame, f'Left EAR: {leftEAR:.2f}', (x - 80, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                # cv2.putText(frame, f'Right EAR: {rightEAR:.2f}', (x + w , y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(frame, "MAR: {:.2f}".format(mar), (10, text_start_y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:           
                    # send signal to close eyes
                    if not headless:
                        cv2.putText(frame, "Eyes Closed!", (10,text_start_y-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)                   
            else:
                # send signal to open eyes 
                COUNTER = 0
                
            # Draw text if mouth is open
            if mar > MOUTH_AR_THRESH:
                # send signal to open mouth 
                if not headless:
                    cv2.putText(frame, "Mouth is Open!", (10,text_start_y-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            #else:
                # send signal to close mouth 
  
    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
