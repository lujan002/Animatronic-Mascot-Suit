'''
import cv2
import dlib
import numpy as np
# Import other necessary libraries

# Initialize Dlib's face detector (HOG-based) and then create the facial landmark predictor

# Define functions to detect blinks, open mouths, and emotions
# These functions will likely rely on the facial landmarks to identify specific points on the face

def detect_blinks(landmarks):
    # Implement blink detection logic based on eye aspect ratio (EAR)
    pass

def detect_open_mouth(landmarks):
    # Implement open mouth detection logic based on the mouth aspect ratio (MAR) or similar metric
    pass

def detect_emotion(face_img):
    # Implement emotion detection, possibly using a pre-trained model or another heuristic
    pass

# Main loop to capture video from webcam
# For each frame:
    # Detect faces
    # For each face:
        # Apply facial landmark detection
        # Call detect_blinks, detect_open_mouth, and detect_emotion with appropriate arguments
        # Display the results (e.g., draw indicators for blinks, open mouths, and show detected emotion)

# Release the video capture object and close windows
'''

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:14:29 2024

@author: 20Jan
"""

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from tensorflow.keras.models import load_model

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

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
    predictions = predictions / sum_predictions if np.any(sum_predictions) else predictions
    return predictions


    # Load your custom model
model = load_model('/Users/20Jan/Junior Jay Capstone/JJ Code/model_optimal2.h5')
# Mapping from your model's output classes to human-readable labels
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Initialize the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize an empty list to hold the largest face
largest_face_list = []

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# This is the path to dlib’s pre-trained facial landmark detector. You can download the detector along with the source code + example videos to this tutorial using the “Downloads” section of the bottom of this blog post.
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# This optional switch controls the path to an input video file residing on disk. If you instead want to work with a live video stream, simply omit this switch when executing the script.
# ap.add_argument("-v", "--video", type=str, default="",
# 	help="path to input video file")
# args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = 0.79

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

# start the video stream thread
# print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
#fileStream = True
# vs = VideoStream(src=0).start() #uncomment if using video stream
# vs = VideoStream(usePiCamera=True).start() #uncomment if using raspberry pi
fileStream = False #uncomment if using video stream
time.sleep(1.0)

# Start video capture
cap = cv2.VideoCapture(0)  # Use 'video3.mp4' for a file instead of 0 for webcam
vs = cap

# loop over frames from the video stream
while True:
	# grab the frame from the video stream, resize
	# it, and convert it to grayscale channels)
# 	frame = vs.read()
	ret, frame = cap.read()
	if not ret:
		break
    
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray_frame = gray
	# detect faces in the grayscale frame
	#rects = detector(gray, 0)
    # Detect faces in the image using Haar Cascade
	faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# 	faces = rects
	rects = faces
	if len(faces) > 0:
        # Focus on the largest face found in the frame
		largest_face = max(faces, key=lambda detection: detection[2] * detection[3])
		largest_face_list = [largest_face]  # Make it a list to use in the for loop
        
	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
          
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
          
        # extract the mouth coordinates, then use the
        # coordinates to compute the mouth aspect ratio
		mouth = shape[mStart:mEnd]
		mouthMAR = mouth_aspect_ratio(mouth)
		mar = mouthMAR
        
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
        
		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # compute the convex hull for the mouth, then
		# visualize the mouth
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		cv2.putText(frame, "MAR: {:.2f}".format(mar), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
        # check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1
		# otherwise, the eye aspect ratio is not below the blink
		# threshold
			cv2.putText(frame, "Eyes are Closed!", (250,60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
			# reset the eye frame counter
			COUNTER = 0
            
        # Draw text if mouth is open
		if mar > MOUTH_AR_THRESH:
			cv2.putText(frame, "Mouth is Open!", (10,60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            
		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (150, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
	for (x, y, w, h) in rects:
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

	# show the frame
	cv2.imshow("Frame", frame)
    
	# Display the resulting frame
	cv2.imshow('Emotion Detection', frame)
    
	# if the `q` key was pressed, break from the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

