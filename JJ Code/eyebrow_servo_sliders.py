# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:04:34 2024

@author: 20Jan
"""


import tkinter as tk
from tkinter import ttk
import serial
import time

# Connect to Arduino
arduino = serial.Serial('COM4', 9600)
time.sleep(2) # wait for the serial connection to initialize

# Function to update the label and send data to Arduino
def update_label(slider, label):
    value = slider.get()
    label.config(text=f"Value: {value}")
    send_to_arduino()

# Function to send all slider values to Arduino
def send_to_arduino():
    value1 = slider1.get()
    value2 = slider2.get()

    # Add more values as needed
    data_string = f"{value1},{value2}\n"  # Format to be parsed by Arduino
    #print(f"Sending to Arduino: {data_string}") # Debugging 
    arduino.write(data_string.encode())

# Create the main window
root = tk.Tk()
root.title("Servo Control")

# Create the first slider
slider1 = ttk.Scale(root, from_=0, to=180, orient='horizontal')
slider1.pack()
slider1.set(90)
label1 = tk.Label(root, text="Value: 0")
label1.pack()
slider1.bind("<Motion>", lambda event: update_label(slider1, label1))

# Bind the adjust_slider2_range function to slider1 movements
# slider1.bind("<Motion>", lambda event: adjust_slider2_range())
# slider1.bind("<ButtonRelease-1>", lambda event: adjust_slider2_range())  # For click and drag


# Create the second slider
slider2 = ttk.Scale(root, from_=0, to=180, orient='horizontal')
slider2.pack()
slider2.set(90) # Set the initial value of the slider to 45 (or any value within the range)
label2 = tk.Label(root, text="Value: 0")
label2.pack()
slider2.bind("<Motion>", lambda event: update_label(slider2, label2))

# Initial call to set slider2's range based on slider1's initial position
# adjust_slider2_range()

# Run the application
root.mainloop()
