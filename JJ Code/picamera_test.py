from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder

def capture_still():
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    
    # Start the camera to allow it to set its automatic settings.
    picam2.start()
    picam2.wait_for_ready()

    # Capture an image. This method returns a capture result object.
    capture_result = picam2.capture_file("test.jpg")
    
    # Stop the camera
    picam2.stop()

    print("Capture complete")

# Call the function to capture an image
capture_still()

