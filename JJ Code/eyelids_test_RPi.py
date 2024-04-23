from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import RPi.GPIO as GPIO
import time

# Define the pin numbers for the limit switches
limitSwitchPin1 = 17
limitSwitchPin2 = 27

# Define the pin number for the servo
servoPin = 22  # Using GPIO18, make sure to use a PWM capable pin

# Setup pigpio for hardware PWM which provides better control
factory = PiGPIOFactory()

# Create a servo object
myServo = Servo(servoPin, pin_factory=factory, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)

# Configure the limit switch pins as input with internal pull-up resistor
GPIO.setup(limitSwitchPin1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(limitSwitchPin2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def initialize_servo():
    # Move servo to initial position
    myServo.min()  # Equivalent to myServo.write(0) in Arduino
    time.sleep(1)  # Allow time for the servo to reach the position

def check_switches_and_control_servo():
    while True:
        stateSwitch1 = GPIO.input(limitSwitchPin1)
        stateSwitch2 = GPIO.input(limitSwitchPin2)
        
        if stateSwitch1 == GPIO.LOW:  # If switch 1 is pressed
            myServo.max()  # Equivalent to myServo.write(180) in Arduino
        elif stateSwitch2 == GPIO.LOW:  # If switch 2 is pressed
            myServo.min()  # Equivalent to myServo.write(0) in Arduino

        time.sleep(0.01)  # Delay to prevent bouncing effects

if __name__ == '__main__':
    try:
        initialize_servo()
        check_switches_and_control_servo()
    except KeyboardInterrupt:
        print("Program terminated")
        GPIO.cleanup()
