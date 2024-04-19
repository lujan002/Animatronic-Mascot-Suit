import RPi.GPIO as GPIO
import time

# Set the GPIO pin numbers the servos are connected to
servo_pin_1 = 26  # Servo 1
servo_pin_2 = 16  # Servo 2

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
    time.sleep(0.02)  # Short delay to allow movement to new position

def sweep_servos():
    # Servo 1 sweeps from 75 to 115, Servo 2 from 115 to 75
    for angle in range(75, 116):  # Increment angle
        set_servo_angles(angle, 190 - angle)  # 190 - angle calculates the opposite movement
        print(f"Servo 1: {angle} degrees, Servo 2: {190 - angle} degrees")

    # Servo 1 sweeps from 115 back to 75, Servo 2 from 75 back to 115
    for angle in range(115, 74, -1):  # Decrement angle
        set_servo_angles(angle, 190 - angle)
        print(f"Servo 1: {angle} degrees, Servo 2: {190 - angle} degrees")

try:
    while True:
        sweep_servos()  # Call the sweep function in a loop

except KeyboardInterrupt:
    # Clean up the GPIO on CTRL+C exit
    pwm1.stop()
    pwm2.stop()
    GPIO.cleanup()

# Clean up the GPIO on normal exit
pwm1.stop()
pwm2.stop()
GPIO.cleanup()
