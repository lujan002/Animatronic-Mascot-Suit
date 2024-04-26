# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:34:46 2024

@author: 20Jan

Program to test movement of the beak. Slider 1 and 2 have an inverse relationship centered around 95 degrees (due to physical calibration) 
Sliders 3 and 4 are irrelevent and just meant for lazy compatibility with control_servo_serial.ino 
"""

import tkinter as tk
from tkinter import ttk
import serial
import time

# Connect to Arduino
arduino = serial.Serial('COM5', 9600)
time.sleep(2) # wait for the serial connection to initialize

# Function to send all slider values to Arduino
def send_to_arduino():
    data_string = f"{slider1.get()},{slider2.get()},{slider3.get()},{slider4.get()}\n"  # Format to be parsed by Arduino
    #print(f"Sending to Arduino: {data_string}") # Debugging 
    arduino.write(data_string.encode())

def update_slider1(event):
    value1 = slider1.get()
    label1.config(text=f"Value: {value1}")
    # Update slider2 in the opposite direction
    slider2.set(178 - value1)
    update_slider2(None)  # Update slider2's label and send to Arduino
    send_to_arduino()

def update_slider2(event):
    value2 = slider2.get()
    label2.config(text=f"Value: {value2}")
    # Update slider1 in the opposite direction only if this wasn't called by slider1's update
    if event:
        slider1.set(178 - value2)
        update_slider1(None)  # Update slider1's label and send to Arduino
    send_to_arduino()

# Create the main window
root = tk.Tk()
root.title("Servo Control")

# LEFT BEAK SERVO IS SERVO 1 (PIN 10)
# Create the first slider
slider1 = ttk.Scale(root, from_=85, to=111, orient='horizontal')
slider1.pack()
slider1.set(40)
label1 = tk.Label(root, text="Value: 40")
label1.pack()
slider1.bind("<B1-Motion>", update_slider1)  # Handle drag motion

# RIGHT BEAK SERVO IS SERVO 2 (PIN 11)
# Create the second slider
slider2 = ttk.Scale(root, from_=67, to=82, orient='horizontal')
slider2.pack()
slider2.set(140)  # Initially set to opposite of slider1
label2 = tk.Label(root, text="Value: 140")
label2.pack()
slider2.bind("<B1-Motion>", update_slider2)  # Handle drag motion

# Create the third slider
slider3 = ttk.Scale(root, from_=0, to=180, orient='horizontal')
slider3.pack()
slider3.set(90)
label3 = tk.Label(root, text="Value: 90")
label3.pack()
slider3.bind("<Motion>", lambda event: update_label(slider3, label3))

# Create the fourth slider
slider4 = ttk.Scale(root, from_=0, to=180, orient='horizontal')
slider4.pack()
slider4.set(50)
label4 = tk.Label(root, text="Value: 50")
label4.pack()
slider4.bind("<Motion>", lambda event: update_label(slider4, label4))

# Function to update the label for slider 3 and 4
def update_label(slider, label):
    value = slider.get()
    label.config(text=f"Value: {value}")
    send_to_arduino()

# Run the application
root.mainloop()
