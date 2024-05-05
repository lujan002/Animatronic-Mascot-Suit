import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Create a Face Detection model
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Start video capture
cap = cv2.VideoCapture('/dev/video0')  # Adjust this for your camera

# Padding parameters
top, bottom, left, right = [100]*4  # Adjust these values based on your needs

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Add padding to the image to simulate a larger field of view
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False

    # Detect faces
    results = face_detection.process(image)

    # Draw face detections
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)

    # Display the resulting frame
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to exit
        break

cap.release()
cv2.destroyAllWindows()
