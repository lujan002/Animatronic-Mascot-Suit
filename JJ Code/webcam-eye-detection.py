# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 08:51:03 2024

@author: 20Jan
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model



# Load your custom model for open/closed mouth detection
model = load_model('/Users/20Jan/Junior Jay Capstone/JJ Code/yawn_model_80.h5')

# Mapping from your model's output classes to human-readable labels
# Update this based on your new model's classes. Assuming 0: 'Closed', 1: 'Open'
mouth_state_dict = {0: 'Closed', 1: 'Open'}

# Initialize the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)  # Use 'video3.mp4' for a file instead of 0 for webcam

from tensorflow.keras.optimizers import Adam
# Assuming your model is for a classification task
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.compile()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop and resize the image to fit model's expected input
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (100, 100))
        roi = roi_gray.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # Make prediction using your model
        prediction = model.predict(roi)
        
        # Get the label of the class with the highest confidence
        mouth_state_label = mouth_state_dict[np.argmax(prediction)]
  
        # Display the label above the rectangle
        cv2.putText(frame, mouth_state_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Mouth State Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
