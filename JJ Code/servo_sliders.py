# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:34:46 2024

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
    value3 = slider3.get()
    value4 = slider4.get()
    # Add more values as needed
    data_string = f"{value1},{value2},{value3},{value4}\n"  # Format to be parsed by Arduino
    #print(f"Sending to Arduino: {data_string}") # Debugging 
    arduino.write(data_string.encode())
    
# Create the main window
root = tk.Tk()
root.title("Servo Control")

# Create the first slider
slider1 = ttk.Scale(root, from_=0, to=180, orient='horizontal')
slider1.pack()
slider1.set(40)
label1 = tk.Label(root, text="Value: 0")
label1.pack()
slider1.bind("<Motion>", lambda event: update_label(slider1, label1))

# Create the second slider
slider2 = ttk.Scale(root, from_=0, to=180, orient='horizontal')
slider2.pack()
slider2.set(90) # Set the initial value of the slider to 45 (or any value within the range)
label2 = tk.Label(root, text="Value: 0")
label2.pack()
slider2.bind("<Motion>", lambda event: update_label(slider2, label2))

# Create the third slider
slider3 = ttk.Scale(root, from_=0, to=180, orient='horizontal')
slider3.pack()
slider3.set(90)
label3 = tk.Label(root, text="Value: 0")
label3.pack()
slider3.bind("<Motion>", lambda event: update_label(slider3, label3))

# Create the second slider
slider4 = ttk.Scale(root, from_=0, to=180, orient='horizontal')
slider4.pack()
slider4.set(50)
label4 = tk.Label(root, text="Value: 0")
label4.pack()
slider4.bind("<Motion>", lambda event: update_label(slider4, label4))

# Run the application
root.mainloop()
