from mtcnn import MTCNN
import cv2


detector = MTCNN()
cap = cv2.VideoCapture('/dev/video0') # 0 for the first camera device

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (MTCNN expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    # Draw bounding boxes around detected faces
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
