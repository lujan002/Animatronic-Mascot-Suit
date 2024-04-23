import tkinter as tk
from tkinter import ttk
import pigpio
import time

# Initialize pigpio library
pi = pigpio.pi()

# GPIO pins where the servos are connected
SERVO_PINS = [5,6,16, 25, 24, 23]

# Function to update the servo position based on slider value
def update_servo(slider, label, pin):
    value = slider.get()
    label.config(text=f"Value: {int(value)}")
    set_servo_angle(pin, value)

# Function to convert angle to PWM pulse width
def set_servo_angle(pin, angle):
    pulse_width = int(angle / 180.0 * 2000 + 500)  # Convert angle to pulse width
    pi.set_servo_pulsewidth(pin, pulse_width)

# Create the main window
root = tk.Tk()
root.title("Servo Control with Raspberry Pi")

# Create sliders and labels for each servo
sliders = []
labels = []
for i, pin in enumerate(SERVO_PINS):
    slider = ttk.Scale(root, from_=0, to=180, orient='horizontal')
    slider.pack()
    slider.set(90)  # Set the initial value of the slider to 90 (middle position)
    label = tk.Label(root, text="Value: 90")
    label.pack()
    sliders.append(slider)
    labels.append(label)
    slider.bind("<Motion>", lambda event, s=slider, l=label, p=pin: update_servo(s, l, p))

# Run the application
root.mainloop()

# Cleanup on exit
for pin in SERVO_PINS:
    pi.set_servo_pulsewidth(pin, 0)  # Turn off servo
pi.stop()
