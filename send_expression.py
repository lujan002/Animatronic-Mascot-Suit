# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:05:20 2024

@author: 20Jan
"""


import serial
import time

# Connect to Arduino
arduino = serial.Serial('COM4', 9600)
time.sleep(2) # wait for the serial connection to initialize

# Function to send expression command to Arduino
def send_expression_to_arduino(expression):
    command = f"{expression}\n"  # Command string to send
    arduino.write(command.encode())

send_expression_to_ardunio

# Run the application
root.mainloop()
