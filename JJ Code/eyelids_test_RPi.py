from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import RPi.GPIO as GPIO
import time

# Define the pin numbers for the limit switches
limitSwtichTop = 17
limitSwtichBottom = 27

# Define the pin number for the servo
servoPin = 22  # Using GPIO18, make sure to use a PWM capable pin

# Setup pigpio for hardware PWM which provides better control
factory = PiGPIOFactory()

# Create a servo object
myServo = Servo(servoPin, pin_factory=factory, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)

# Configure the limit switch pins as input with internal pull-up resistor
GPIO.setup(limitSwtichTop, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(limitSwtichBottom, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def initialize_servo():
    # Move servo to initial position
    myServo.min()  # Equivalent to myServo.write(0) in Arduino
    time.sleep(1)  # Allow time for the servo to reach the position

def check_switches_and_control_servo():
    while True:
        stateSwitchTop = GPIO.input(limitSwtichTop)
        stateSwitchBottom = GPIO.input(limitSwtichBottom)
        
        if stateSwitchTop == GPIO.LOW:  # If switch 1 is pressed
            myServo.max()  # Moves eyes down. Equivalent to myServo.write(180) in Arduino     

        elif stateSwitchBottom == GPIO.LOW:  # If switch 2 is pressed
            continue
            # myServo.min()  # Moves eyes up. Equivalent to myServo.write(0) in Arduino
           
        time.sleep(0.01)  # Delay to prevent bouncing effects

# def check_switches_and_control_servo():
#     stateSwitchTop = GPIO.input(limitSwtichTop)
#     stateSwitchBottom = GPIO.input(limitSwtichBottom)
    
#     # if stateSwitchTop == GPIO.LOW:  # If switch 1 is pressed
#     myServo.max()  # Moves eyes down. Equivalent to myServo.write(180) in Arduino     

#     if stateSwitchBottom == GPIO.LOW:  # If switch 2 is pressed
#         time.sleep(5)
#         # myServo.min()  # Moves eyes up. Equivalent to myServo.write(0) in Arduino
        
#     time.sleep(0.01)  # Delay to prevent bouncing effects

# def check_switches_and_control_servo():
#     # Define the servo direction; True for upward, False for downward
#     direction_up = True
#     # Time interval for switching
#     switch_interval = 1
    
#     # Initial time to start counting
#     last_switch_time = time.time()
    
#     while True:
#         current_time = time.time()
#         stateSwitchTop = GPIO.input(limitSwtichTop)
#         stateSwitchBottom = GPIO.input(limitSwtichBottom)
        
#         # Check if it's time to switch direction due to time interval
#         if current_time - last_switch_time >= switch_interval:
#             direction_up = not direction_up
#             last_switch_time = current_time
        
#         # Check if the top switch is pressed
#         if stateSwitchTop == GPIO.LOW:
#             if not direction_up:  # Only change direction if currently moving down
#                 direction_up = True
#                 last_switch_time = current_time  # Reset timer to delay next switch

#         # Check if the bottom switch is pressed
#         elif stateSwitchBottom == GPIO.LOW:
#             if direction_up:  # Only change direction if currently moving up
#                 direction_up = False
#                 last_switch_time = current_time  # Reset timer to delay next switch

#         # Move the servo based on the direction
#         if direction_up:
#             myServo.max()  # Move servo upward
#         else:
#             myServo.min()  # Move servo downward
        
#         time.sleep(0.01)  # Delay to prevent bouncing effects and reduce CPU usage
        
if __name__ == '__main__':
    try:
        initialize_servo()
        check_switches_and_control_servo()
    except KeyboardInterrupt:
        print("Program terminated")
        GPIO.cleanup()
