from picamera2 import Picamera2
import time

def capture_and_save_image():
    # Create a Picamera2 object
    picam2 = Picamera2()
    
    # Configure the camera
    picam2.start_preview()
    time.sleep(2)  # Wait for 2 seconds to allow the camera to adjust to lighting conditions
    
    try:
        # Capture and save an image
        image = picam2.capture_image()
        with open("captured_image.jpg", "wb") as img_file:
            img_file.write(image.as_rgb())
        print("Image has been captured and saved as captured_image.jpg")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    capture_and_save_image()
