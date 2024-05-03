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

from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory

import mediapipe as mp

# Initialize pigpio
pi = pigpio.pi()

# # Constants for GPIO pin numbers for the servos
beak_l = 5 # Left beak
beak_r = 6 # Right beak
eyelid_l = 22# Left eyelid servo
eyelid_l_sw_t = 17#Left eyelid top limit switch
eyelid_l_sw_b = 27#Left eyelid bottom limit switch
eyelid_r = 18# Right eyelid servo
eyelid_r_sw_t = 13# Right eyelid top limit switch
eyelid_r_sw_b = 12# Right eyelid bottom limit switch
# # angry_led =
# # happy_led = 
# # sad_led = 
# # surprised_led =
# # neutral_led =
# # mouth_led = 
# # eye_led = 

# # Setup pigpio for hardware PWM which provides better control
# factory = PiGPIOFactory()

# # Create a servo object
# L_Eyelid_Servo = Servo(eyelid_l, pin_factory=factory, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

# # Set up limit switch pins as an input and enable the internal pull-up resistor
# GPIO.setup(eyelid_l_sw_b, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Constants for eyebrow control
eyebrow_angles = {
    'Angry': {'l_o': 60, 'l_i': 63, 'r_i': 108, 'r_o': 145},
    'Happy': {'l_o': 121, 'l_i': 82, 'r_i': 35, 'r_o': 108},
    'Sad': {'l_o': 148, 'l_i': 127, 'r_i': 63, 'r_o': 82},
    'Surprise': {'l_o': 45, 'l_i': 150, 'r_i': 25, 'r_o': 137},
    'Neutral': {'l_o': 90, 'l_i': 100, 'r_i': 90, 'r_o': 113},
    'Rizz': {'l_o': 95, 'l_i': 71, 'r_i': 29, 'r_o': 92}
}

L_BEAK_OPEN_ANGLE = 111
R_BEAK_OPEN_ANGLE = 67
L_BEAK_CLOSED_ANGLE = 88
R_BEAK_CLOSED_ANGLE = 90

# GPIO pin numbers for the servos
eyebrow_pins = {
    'l_o': 16,  # Example GPIO pin numbers
    'l_i': 25,
    'r_i': 24,
    'r_o': 23
}

# # Configure the limit switch pins as input with internal pull-up resistor
GPIO.setmode(GPIO.BCM)
GPIO.setup(eyelid_l_sw_t, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(eyelid_l_sw_b, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(eyelid_r_sw_t, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(eyelid_r_sw_b, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Setup pigpio for hardware PWM which provides better control
factory = PiGPIOFactory()

# Create a servo object
eyelid_l_servo = Servo(eyelid_l, pin_factory=factory, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
eyelid_r_servo = Servo(eyelid_r, pin_factory=factory, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

def set_eyebrow(emotion):
    # Check if the emotion is in the predefined angles
    if emotion in eyebrow_angles:
        # Set each eyebrow servo to the angle based on the emotion
        for key, pin in eyebrow_pins.items():
            angle = eyebrow_angles[emotion][key]
            set_servo_angle(pin, angle)
            print(f"Set eyebrow '{key}' to angle {angle} for emotion '{emotion}'.")
    else:
        print(f"No eyebrow angles defined for emotion '{emotion}'.")

# Helper function to set servo angles
def set_servo_angle(servo_pin, angle):
    # Calculate PWM duty cycle for the given angle
    pulsewidth = int((angle / 180.0) * 2000 + 500)
    pi.set_servo_pulsewidth(servo_pin, pulsewidth)

def control_eyelid(servo_pin, direction, simulate_limit_switch=False):
    try:
        pulsewidth = 1500  # Neutral position
        if direction == 'close':
            pulsewidth = 2000  # Rotate clockwise
            limit_switch = eyelid_l_sw_b if servo_pin == eyelid_l else eyelid_r_sw_b
        elif direction == 'open':
            pulsewidth = 1000  # Rotate counterclockwise
            limit_switch = eyelid_l_sw_t if servo_pin == eyelid_l else eyelid_r_sw_t
        else:
            raise ValueError("Invalid direction for eyelid control: must be 'close' or 'open'")

        # Command the servo to move
        pi.set_servo_pulsewidth(servo_pin, pulsewidth)
        time.sleep(0.01)  # Give some time for servo to reach the position

        # Simulate or check limit switch
        if simulate_limit_switch:
            print("Simulation mode: Assuming limit switch reached, stopping servo.")
            pi.set_servo_pulsewidth(servo_pin, 1500)  # Stop the servo
        else:
            if GPIO.input(limit_switch) == GPIO.LOW:
                print("Limit switch reached, stopping servo.")
                pi.set_servo_pulsewidth(servo_pin, 1500)  # Stop the servo
            else:
                print("Limit switch not reached, checking again.")
                time.sleep(0.2)  # Give more time and check again
                if GPIO.input(limit_switch) == GPIO.LOW:
                    print("Limit switch reached after additional time, stopping servo.")
                    pi.set_servo_pulsewidth(servo_pin, 1500)  # Stop the servo
                else:
                    print("Warning: Limit switch not reached. Possible mechanical issue or misalignment.")
    except Exception as e:
        print(f"Error controlling eyelid: {e}")


# Define whether to run the script in headless mode
headless = False

# Load TFLite model and allocate tensors (for emotion detection).
# interpreter = tf.lite.Interpreter(model_path="/Users/20Jan/Junior Jay Capstone/JJ Code/model.tflite")
interpreter = tf.lite.Interpreter(model_path="/home/lujan002/Repositories/Animatronic-Mascot-Suit/JJ Code/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Constants for model input and output
input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']

emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Bias factor adjustments
ANG_INDEX = 0
HAP_INDEX = 3
SAD_INDEX = 5  
SUR_INDEX = 6
NEU_INDEX = 4
ANG_BIAS_FACTOR = 0.25
HAP_BIAS_FACTOR = 0.25
SAD_BIAS_FACTOR = 1
NEU_BIAS_FACTOR = 0.5
SUR_BIAS_FACTOR = 1

# Load dlib model for eye and mouth detection
predictor = dlib.shape_predictor('/home/lujan002/Repositories/Animatronic-Mascot-Suit/JJ Code/shape_predictor_68_face_landmarks.dat')

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Create a Face Detection model
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Constants for blink and mouth open detection
EYE_AR_THRESH = 0.21
MOUTH_AR_THRESH = 0.80
EYE_AR_CONSEC_FRAMES = 3
MOUTH_OPEN_AR_CONSEC_FRAMES = 4
MOUTH_CLOSED_AR_CONSEC_FRAMES = 4

# initialize the frame counters and the total number of blinks
EYE_COUNTER = 0
MOUTH_OPEN_COUNTER = 0
MOUTH_CLOSED_COUNTER = 0

# Initialize emotion stability control
emotion_stability_buffer = []
stability_frames = 3  # Number of frames to keep emotion stable
stable_emotion = ["Neutral"] * stability_frames


# Function to adjust predictions
def adjust_predictions(predictions):
    print("Before adjustment:", predictions)
    predictions[0][ANG_INDEX] *= ANG_BIAS_FACTOR
    predictions[0][HAP_INDEX] *= HAP_BIAS_FACTOR
    predictions[0][SAD_INDEX] *= SAD_BIAS_FACTOR
    predictions[0][SUR_INDEX] *= SUR_BIAS_FACTOR
    predictions[0][NEU_INDEX] *= NEU_BIAS_FACTOR
    total = np.sum(predictions)
    predictions /= total if total > 0 else 1
    print("After adjustment:", predictions)
    return predictions

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

def remove_classes(predictions, classes_to_remove):
    predictions[:, classes_to_remove] = 0
    sum_predictions = np.sum(predictions, axis=1, keepdims=True)
    return predictions / sum_predictions if np.any(sum_predictions) else predictions

#GPIO Basic initialization
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Use a variable for the Pin to use
pin_mapping = {
    'Angry': 4,
    'Happy': 5,
    'Sad': 6,
    'Surprise': 7,
    'Neutral': 8,
    'Eyes': 9,
    'Mouth': 10
}

# Initialize your pins
for pin in pin_mapping.values():
    GPIO.setup(pin, GPIO.OUT)

def turn_led(emotion, on=True):
    """Control LEDs based on emotion and state."""
    pin = pin_mapping.get(emotion)
    if pin is None:
        print(f"No GPIO pin assigned for the emotion: {emotion}")
        return
    GPIO.output(pin, on)
    print(f"{'Turning on' if on else 'Turning off'} {emotion} LED.")
    
# Start video capture
cap = cv2.VideoCapture('/dev/video0')
# cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open the camera.")
else:
    print("Camera is successfully opened.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Ensure the settings were applied by checking the actual values
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera set to width: {actual_width}, height: {actual_height}")

# Padding parameters
top, bottom, left, right = [100]*4  # Adjust these values based on your needs

frame_rate_limit = 5 # Target number of frames per second
last_time = time.time()

while True:   
    # frame_count += 1
    # if frame_count % frame_skip != 0:
    #     continue
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not read.")
        break
    # Alternative approach: Capture less frames
    current_time = time.time()
    elapsed = current_time - last_time

    if elapsed < 1.0 / frame_rate_limit:
        continue  # Skip processing this frame
    last_time = current_time

    # Add padding to the image to simulate a larger field of view
    frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Convert the BGR image to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    frame.flags.writeable = False
    # Detect faces
    results = face_detection.process(frame)
    # Draw face detections
    frame.flags.writeable = True
    # Convert back to BGR for normal viewing
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 

    if not results.detections:
        print("No faces detected.")
    
    if results.detections:
        try:
            # Find the most prominent face based on confidence or another criterion
            chosen_face = max(results.detections, key=lambda det: det.score)
            if chosen_face is None:
                print("No prominent face detected despite detections being present.")
                continue
            # Extract bounding box coordinates from the chosen face
            bboxC = chosen_face.location_data.relative_bounding_box
            if bboxC is None:
                print("Bounding box is None.")
                continue
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            if bbox[2] <= 0 or bbox[3] <= 0:
                print("Invalid bounding box dimensions.")
                continue
            x, y, w, h = bbox

            # Convert to greyscale 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Plot the border around the face
            if not headless:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize the regionFalse of interest for emotion recognition
            roi = cv2.resize(face_roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = roi.astype(np.float32)  # Ensure the data type is float32 for tflite
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            
            # Use the TFLite model
            interpreter.set_tensor(input_details[0]['index'], roi)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
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

            if not headless:
                # Put the emotion label above the rectangle
                cv2.putText(frame, stable_emotion[-1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                # Optionally, show all emotions with their probabilities
                text_start_y = y + h + 20
                for i, prob in enumerate(predictions[0]):
                    if i not in [1, 2]:  # Only process if not in the ignore list
                        emotion_label = emotion_dict[i]
                        if prob > 0:  # Only display if probability is not zero
                            cv2.putText(frame, f"{emotion_label}: {prob:.2f}", (x + 10, text_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
                            text_start_y += 15  # Move to the next line for the next label
                    
            # Send GPIO signal after deciding emotion 
            if stable_emotion[-1] != stable_emotion[-2]:
                turn_led(stable_emotion[-2], False)
                turn_led(stable_emotion[-1])
                # send signal to update eyebrows based on emotion   
                set_eyebrow(stable_emotion[-1])

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
            
            if not headless:
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
                # cv2.putText(frame, f'Left EAR: {leftEAR:.2f}', (x - 80, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                # cv2.putText(frame, f'Right EAR: {rightEAR:.2f}', (x + w , y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(frame, "MAR: {:.2f}".format(mar), (x+w, y+70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                EYE_COUNTER += 1
                if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # otherwise, the eye aspect ratio is not below the blink threshold
                    turn_led('Eyes')                
                    # send signal to close eyes
                    # L_Eyelid_Servo.max()  # Moves eyes down. Equivalent to myServo.write(0) in Arduino
                    # if GPIO.input(eyelid_l_sw_b) == GPIO.LOW:  # If bottom switch is pressed
                    #     continue
                    EYE_COUNTER = 0

                    if not headless:
                        cv2.putText(frame, "Eyes Closed!", (x+w,y+40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)                   
            else:
                turn_led('Eyes', False)
                # send signal to open eyes 
                # L_Eyelid_Servo.min()  # Moves eyes up. Equivalent to myServo.write(0) in Arduino
                # if GPIO.input(eyelid_l_sw_b) == GPIO.LOW:  # If switch 2 is pressed
                #     continue
                EYE_COUNTER = 0
                
            # Draw text if mouth is open
            if mar > MOUTH_AR_THRESH:
                MOUTH_OPEN_COUNTER += 1
                if MOUTH_OPEN_COUNTER > MOUTH_OPEN_AR_CONSEC_FRAMES:
                    turn_led('Mouth')
                    # send signal to open mouth 
                    set_servo_angle(beak_l, L_BEAK_OPEN_ANGLE)
                    set_servo_angle(beak_r, R_BEAK_OPEN_ANGLE)
                    if not headless:
                        cv2.putText(frame, "Mouth is Open!", (x+w,y+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                    MOUTH_OPEN_COUNTER = 0
            else:
                MOUTH_CLOSED_COUNTER += 1
                if MOUTH_CLOSED_COUNTER > MOUTH_CLOSED_AR_CONSEC_FRAMES:
                    turn_led('Mouth', False)
                    # send signal to close mouth 
                    set_servo_angle(beak_l, L_BEAK_CLOSED_ANGLE)
                    set_servo_angle(beak_r, R_BEAK_CLOSED_ANGLE)
                    MOUTH_CLOSED_COUNTER = 0
        except Exception as e:
            print(f"Error processing the most prominent face: {e}")
    if not headless:
        # Display the resulting frame
        cv2.imshow('Face and Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



