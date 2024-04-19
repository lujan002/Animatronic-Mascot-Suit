import RPi.GPIO as GPIO
import time

# Set the GPIO pin numbers the servos are connected to
servo_pin_1 = 17  # Servo 1
servo_pin_2 = 26  # Servo 2

# Setup the GPIO pins for output
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin_1, GPIO.OUT)
GPIO.setup(servo_pin_2, GPIO.OUT)

# Set up PWM on the pins, Pulse Width Modulation Frequency
pwm1 = GPIO.PWM(servo_pin_1, 50)  # Frequency is 50Hz, which is typical for servos
pwm2 = GPIO.PWM(servo_pin_2, 50)  # Frequency is 50Hz, which is typical for servos

# Start PWM running, but with value of 0 (pulse off)
pwm1.start(0)
pwm2.start(0)

def set_servo_angles(angle1, angle2):
    duty_cycle_1 = angle1 / 18.0 + 2
    duty_cycle_2 = angle2 / 18.0 + 2
    pwm1.ChangeDutyCycle(duty_cycle_1)
    pwm2.ChangeDutyCycle(duty_cycle_2)
    time.sleep(0.5)  # Longer delay to ensure servo reaches position

# Set servos to a static position
static_angle_1 = 95  # Static angle for Servo 1
static_angle_2 = 95  # Static angle for Servo 2
set_servo_angles(static_angle_1, static_angle_2)
print(f"Servo 1 set to {static_angle_1} degrees, Servo 2 set to {static_angle_2} degrees")

try:
    # Keep the program running to maintain the servo positions
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    # Clean up the GPIO on CTRL+C exit
    pwm1.stop()
    pwm2.stop()
    GPIO.cleanup()

# Clean up the GPIO on normal exit
pwm1.stop()
pwm2.stop()
GPIO.cleanup()
