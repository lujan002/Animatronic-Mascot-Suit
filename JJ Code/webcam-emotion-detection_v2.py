# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:24:27 2024

@author: 20Jan
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:48:18 2024

@author: 20Jan
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

def remove_classes(predictions, classes_to_remove):
    predictions[:, classes_to_remove] = 0
    sum_predictions = np.sum(predictions, axis=1, keepdims=True)
    predictions = predictions / sum_predictions if np.any(sum_predictions) else predictions
    return predictions


# Load your custom model
model = load_model('/Users/20Jan/Junior Jay Capstone/JJ Code/model_optimal2.h5')
# model 2 = 
# model 3 = 
# Mapping from your model's output classes to human-readable labels
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Initialize the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize an empty list to hold the largest face
largest_face_list = []

# Start video capture
cap = cv2.VideoCapture(0)  # Use 'video3.mp4' for a file instead of 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale as the Haar Cascade requires gray images
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image using Haar Cascade
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Focus on the largest face found in the frame
        largest_face = max(faces, key=lambda detection: detection[2] * detection[3])
        largest_face_list = [largest_face]  # Make it a list to use in the for loop
        
    for (x, y, w, h) in largest_face_list:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Get the region of interest from the grayscale frame
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to match your model's expected input
        roi = roi_gray.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # Predict emotion
        prediction = model.predict(roi)
        # Predict open eye
        # prediction = model2.predict(roi)
        # Predict open mouth
        # prediction = model3.predict(roi)
        
        # Assuming 'fear' is at index 2 and 'disgust' is at index 1
        classes_to_remove = [1, 2]
        new_predictions = remove_classes(prediction, classes_to_remove)
        
        emotion_label = emotion_dict[np.argmax(new_predictions)]

        # Put the emotion label above the rectangle
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


