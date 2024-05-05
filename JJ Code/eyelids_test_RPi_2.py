from gpiozero import Servo, Button
from time import sleep

# Define the GPIO pin for the servo and the limit switch
servo_pin = 18 # Change this to the pin connected to your servo
limit_switch_pin = 13  # Change this to the pin connected to your limit switch

# Create a Servo object
servo = Servo(servo_pin)

# Create a Button object for the limit switch
limit_switch = Button(limit_switch_pin)

def move_servo(direction='forward'):
    if direction not in ['forward', 'backward']:
        print("Invalid direction specified. Please use 'forward' or 'backward'.")
        return
    
    print(f"Moving servo {direction}...")
    if direction == 'forward':
        servo.forward()
    elif direction == 'backward':
        servo.backward()

    # Wait until the limit switch is pressed
    limit_switch.wait_for_press()
    print("Limit switch activated. Stopping servo.")
    
    # Stop the servo
    servo.stop()

# Example usage: Move the servo forward
move_servo('forward')

# Example usage: Move the servo backward
# move_servo('backward')
