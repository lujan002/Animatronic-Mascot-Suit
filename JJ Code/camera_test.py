import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera could not be opened.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab a frame")
            break
        cv2.imshow("Test Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
