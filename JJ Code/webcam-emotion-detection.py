# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 09:05:21 2024

@author: 20Jan
"""


import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

def remove_classes(predictions, classes_to_remove):
    # Set the predictions for unwanted classes to 0
    predictions[:, classes_to_remove] = 0
    # Normalize the predictions so they sum to 1
    sum_predictions = np.sum(predictions, axis=1, keepdims=True)
    predictions = predictions / sum_predictions if np.any(sum_predictions) else predictions
    return predictions

def adjust_thresholds(predictions, thresholds):
    """
    Adjust predictions based on custom thresholds for each class.
    predictions: Numpy array of shape (num_samples, num_classes) with model predictions.
    thresholds: List or numpy array of shape (num_classes,) with custom thresholds.
    """
    adjusted_predictions = (predictions > thresholds).astype(int)
    return adjusted_predictions

# Load your trained model
model = load_model('/Users/20Jan/Junior Jay Capstone/JJ Code/model_optimal2.h5')

# Dictionary mapping the numeric classes to human-readable labels
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
thresholds = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # Custom thresholds for each class

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale (if your model expects grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the grayscale frame to match your model's expected input size
    resized_frame = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

    # Prepare the frame for the model (reshape and scale pixel values)
    x = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
    x = np.expand_dims(x, axis=-1)  # Add channel dimension
    x = x / 255.0  # Scale pixel values to [0,1]

    # Make prediction
    prediction = model.predict(x)
    
    # Assuming 'fear' is at index 2 and 'disgust' is at index 1
    classes_to_remove = [1, 2]
    new_predictions = remove_classes(prediction, classes_to_remove)
    adjusted_predictions = adjust_thresholds(new_predictions, thresholds)

    # Proceed with the new_predictions as before
    max_index = np.argmax(new_predictions[0])

    # Get the label corresponding to the highest prediction score
    emotion_label = label_dict[max_index]

    # Display the label on the frame
    cv2.putText(frame, emotion_label, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
