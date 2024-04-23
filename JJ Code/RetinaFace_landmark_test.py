import cv2
import insightface
from insightface.app import FaceAnalysis

def setup_retinaface():
    app = FaceAnalysis()
    app.prepare(ctx_id=-1, nms=0.4)  # ctx_id=-1 for CPU, >=0 for GPU (the GPU number)
    return app

def process_video(video_path):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Load RetinaFace
    retinaface = setup_retinaface()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = retinaface.get(rgb_frame)

        # Draw results on the frame
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            landmarks = face.landmark_2d_106.astype(int)
            for mark in landmarks:
                cv2.circle(frame, (mark[0], mark[1]), 1, (0, 0, 255), -1)

        # Display the resulting frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video('/dev/video0')  # 0 for the default camera
