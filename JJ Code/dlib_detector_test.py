import dlib
import cv2

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Open the default camera
cap = cv2.VideoCapture('/dev/video0')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray, 0)  # The '1' here means to upsample the image once, which can help detect smaller faces

    # Draw rectangles around each face
    for face in faces:
        x, y = face.left(), face.top()
        w, h = face.right() - x, face.bottom() - y
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press 'q' on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
