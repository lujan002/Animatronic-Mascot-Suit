# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:38:20 2024

@author: 20Jan

Analyze eyebrow position to send expressions to arduino
"""

import cv2
import dlib
import numpy as np
import serial
import time

# Initialize dlib's face detector and load the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/juniorjay/Documents/Animatronic Mascot Suit/JJ Code/shape_predictor_68_face_landmarks.dat")

# Establish serial connection with Arduino
#arduino = serial.Serial('COM4', 9600)
#time.sleep(2)  # Wait for connection to establish

def send_expression_to_arduino(expression):
    command = f"{expression}\n"
    arduino.write(command.encode())

def analyze_eyebrows(shape):
    # Implement your logic here to analyze eyebrow positions
    # You can use the positions of the eyebrow landmarks (shape.part(i)) to determine their positions
    # This function should return "normal", "angry", "sad", or "surprised"
    angry = is_angry(shape)
    if angry:  # This is equivalent to if angry == True:
        return "angry"
    else:
        return "normal"  # Return the string "normal" as a placeholder

def calculate_distances(shape):
    # Get the points for the inner and outer eyebrow edges
    left_inner_brow = shape.part(21)
    right_inner_brow = shape.part(22)
    left_outer_brow = shape.part(17)
    right_outer_brow = shape.part(26)
    
    # Calculate distances
    inner_brow_distance = point_distance(left_inner_brow, right_inner_brow)
    left_brow_height = point_distance(left_inner_brow, left_outer_brow)
    right_brow_height = point_distance(right_inner_brow, right_outer_brow)
    
    return inner_brow_distance, left_brow_height, right_brow_height

def point_distance(point1, point2):
    return ((point1.x - point2.x)**2 + (point1.y - point2.y)**2)**0.5
    

def calculate_normalized_distances(shape, inner_brow_distance, left_brow_height, right_brow_height):
    # Points for the corners of the eyes
    left_eye_outer = shape.part(36)
    right_eye_outer = shape.part(45)
    
    # Calculate the reference distance (eye distance)
    eye_distance = point_distance(left_eye_outer, right_eye_outer)

    # Calculate normalized distances
    normalized_inner_brow_distance = inner_brow_distance / eye_distance
    normalized_left_brow_height = left_brow_height / eye_distance
    normalized_right_brow_height = right_brow_height / eye_distance

    return normalized_inner_brow_distance, normalized_left_brow_height, normalized_right_brow_height

def is_angry(shape):
    # Calculate distances
    inner_brow_distance, left_brow_height, right_brow_height = calculate_distances(shape)

    # Normalize distances
    normalized_inner_brow_distance, normalized_left_brow_height, normalized_right_brow_height = calculate_normalized_distances(shape, inner_brow_distance, left_brow_height, right_brow_height)

    # Now you can display the normalized distances instead of the absolute distances
    cv2.putText(frame, f"Norm Inner Brow Dist: {normalized_inner_brow_distance:.2f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # cv2.putText(frame, f"Norm Left Brow Height: {normalized_left_brow_height:.2f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # cv2.putText(frame, f"Norm Right Brow Height: {normalized_right_brow_height:.2f}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Define thresholds for "angry" eyebrows
    distance_threshold = 0.24  # This threshold may need to be adjusted for your specific use case
    # height_threshold = 0.5     # This threshold may need to be adjusted for your specific use case

    # Check if inner brows are lower than the outer brows and are closer together
    # if (normalized_left_brow_height < height_threshold and normalized_right_brow_height < height_threshold and
    #         normalized_inner_brow_distance < distance_threshold):
    if (normalized_inner_brow_distance < distance_threshold):
        return True
    else:
        return False


# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    for face in faces:
        # Get the landmarks/parts for the face.
        shape = predictor(gray, face)
        
        # Draw dots for each eyebrow landmark
        for i in range(17, 27):  # Loop over the eyebrow points
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Draw a green dot at each point
                 
        # Analyze the eyebrow positions to determine the expression
        expression = analyze_eyebrows(shape)
        
        # Send the determined expression to Arduino
        #send_expression_to_arduino(expression)
        
        # Optional: Display the result on the frame
        cv2.putText(frame, expression, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
arduino.close()
