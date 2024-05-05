import RPi.GPIO as GPIO
import time

# Set up GPIO using BCM numbering
GPIO.setmode(GPIO.BCM)

# Define the GPIO pin where the limit switch is connected
limitSwitchPin = 27

# Set up the pin as an input and enable the internal pull-up resistor
GPIO.setup(limitSwitchPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

try:
    while True:
        # Read the state of the limit switch
        if GPIO.input(limitSwitchPin) == GPIO.LOW:
            print("Switch Pressed")
        else:
            print("Switch Released")
        time.sleep(0.5)  # Delay to make the output readable
except KeyboardInterrupt:
    print("Program stopped")
finally:
    GPIO.cleanup()  # Clean up GPIO to ensure all pins are reset properly
