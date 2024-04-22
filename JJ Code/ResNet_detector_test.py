import cv2
import numpy as np

# Path to model files
model_configuration = "/home/lujan002/Repositories/Animatronic-Mascot-Suit/JJ Code/deploy.prototxt"
model_weights = "/home/lujan002/Repositories/Animatronic-Mascot-Suit/JJ Code/res10_300x300_ssd_iter_140000.caffemodel"

# Load the model
net = cv2.dnn.readNetFromCaffe(model_configuration, model_weights)

# Start video capture
cap = cv2.VideoCapture('/dev/video0')  # 0 is typically the default value for the first camera

frame_count = 0
input_size = (300, 300)  # You can change this to other sizes like (224, 224)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 5 != 0:  # Skip every other frame
        continue
    (h, w) = frame.shape[:2]
    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, input_size), scalefactor=1.0, size=input_size, mean=(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
