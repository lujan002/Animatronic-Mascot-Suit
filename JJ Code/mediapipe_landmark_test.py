import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Video capture
cap = cv2.VideoCapture('/dev/video0')
if not cap.isOpened():
    print("Failed to open camera.")
else:
    print("Camera opened successfully.")

# Padding parameters
top, bottom, left, right = [100]*4  # Adjust these values based on your needs

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera.")
        continue
    # Add padding to the image to simulate a larger field of view
    frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = face_mesh.process(rgb_frame)
    
    # Draw the face mesh annotations on the frame.
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw facial landmarks here if needed
            # Example to draw landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow('MediaPipe FaceMesh', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
