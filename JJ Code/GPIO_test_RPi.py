import pigpio
import time

# Constants for GPIO pin numbers
servo_pin_1 = 26  # Servo 1
servo_pin_2 = 16  # Servo 2

# Initialize pigpio
pi = pigpio.pi()

# Helper function to set servo angles
def set_servo_angle(servo_pin, angle):
    # Calculate PWM duty cycle for the given angle
    pulsewidth = int((angle / 180.0) * 2000 + 500)  # Corrected formula
    if pulsewidth < 500:
        pulsewidth = 500  # Ensure pulsewidth is not below 500
    elif pulsewidth > 2500:
        pulsewidth = 2500  # Ensure pulsewidth is not above 2500
    pi.set_servo_pulsewidth(servo_pin, pulsewidth)

# Function to sweep servos
def sweep_servos():
    # Sweep from 95 to 105 degrees
    for angle in range(92, 108):
        set_servo_angle(servo_pin_1, angle)
        set_servo_angle(servo_pin_2, 190 - angle)  # Opposite movement
        print(f"Servo 1: {angle} degrees, Servo 2: {190 - angle} degrees")
        time.sleep(0.02)  # Delay between steps

    time.sleep(3)  # Wait at open position

    # Sweep back from 105 to 95 degrees
    for angle in range(108, 92, -1):
        set_servo_angle(servo_pin_1, angle)
        set_servo_angle(servo_pin_2, 190 - angle)
        print(f"Servo 1: {angle} degrees, Servo 2: {190 - angle} degrees")
        time.sleep(0.02)  # Delay between steps

    time.sleep(3)  # Wait at closed position

# Run the sweeping function in a loop
try:
    while True:
        sweep_servos()

except KeyboardInterrupt:
    # Stop PWM and clean up GPIO on CTRL+C exit
    pi.set_servo_pulsewidth(servo_pin_1, 0)  # Stop servo pulse
    pi.set_servo_pulsewidth(servo_pin_2, 0)
    pi.stop()
