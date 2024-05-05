# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:34:46 2024

Program to test movement of the beak using Raspberry Pi GPIO.
Slider 1 and 2 have an inverse relationship centered around 95 degrees (due to physical calibration)
Sliders 3 and 4 are irrelevant and just meant for lazy compatibility with a previous Arduino version.
"""

import tkinter as tk
from tkinter import ttk
from gpiozero import Servo
from time import sleep

# Define GPIO pins for Servos
servo1_pin = 5  # GPIO pin 17 for Servo 1
servo2_pin = 6 # GPIO pin 18 for Servo 2

# Setup Servo devices
servo1 = Servo(servo1_pin)
servo2 = Servo(servo2_pin)

def update_servos():
    # Convert slider values from scale to -1 to 1 for Servo
    servo1_value = slider_to_servo(slider1.get(), 96, 111)
    servo2_value = slider_to_servo(slider2.get(), 67, 82)
    servo1.value = servo1_value
    servo2.value = servo2_value
    sleep(0.1)  # Add a small delay to allow servo to move

def slider_to_servo(value, min_val, max_val):
    """Convert slider value to -1 to 1 scale for gpiozero Servo."""
    return (value - min_val) / (max_val - min_val) * 2 - 1

def update_slider1(event):
    value1 = slider1.get()
    label1.config(text=f"Value: {value1}")
<<<<<<< HEAD
    slider2.set(178 - value1)
    update_slider2(None)
    update_servos()
=======
    # Update slider2 in the opposite direction
    slider2.set(178 - value1)
    update_slider2(None)  # Update slider2's label and send to Arduino
    send_to_arduino()
>>>>>>> 35ba36025ccfcfcd19106c84bc3b9c439ef26831

def update_slider2(event):
    value2 = slider2.get()
    label2.config(text=f"Value: {value2}")
    if event:
        slider1.set(178 - value2)
<<<<<<< HEAD
        update_slider1(None)
    update_servos()
=======
        update_slider1(None)  # Update slider1's label and send to Arduino
    send_to_arduino()
>>>>>>> 35ba36025ccfcfcd19106c84bc3b9c439ef26831

root = tk.Tk()
root.title("Servo Control")

<<<<<<< HEAD
=======
# LEFT BEAK SERVO IS SERVO 1 (PIN 10)
# Create the first slider
>>>>>>> 35ba36025ccfcfcd19106c84bc3b9c439ef26831
slider1 = ttk.Scale(root, from_=85, to=111, orient='horizontal')
slider1.pack()
slider1.set(98)  # Initial value adjusted
label1 = tk.Label(root, text="Value: 98")
label1.pack()
slider1.bind("<B1-Motion>", update_slider1)

<<<<<<< HEAD
=======
# RIGHT BEAK SERVO IS SERVO 2 (PIN 11)
# Create the second slider
>>>>>>> 35ba36025ccfcfcd19106c84bc3b9c439ef26831
slider2 = ttk.Scale(root, from_=67, to=82, orient='horizontal')
slider2.pack()
slider2.set(80)  # Initially set to opposite of slider1
label2 = tk.Label(root, text="Value: 80")
label2.pack()
slider2.bind("<B1-Motion>", update_slider2)

root.mainloop()
