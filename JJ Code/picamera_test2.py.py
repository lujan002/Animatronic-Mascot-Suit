import subprocess
import os

def capture_image(image_path):
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        # Command to capture an image
        cmd = ["libcamera-still", "-o", image_path]

        # Run the command
        subprocess.run(cmd, check=True)
        print(f"Image captured successfully and saved to {image_path}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while capturing the image: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
capture_image("/home/lujan002/Desktop/image.jpg")
