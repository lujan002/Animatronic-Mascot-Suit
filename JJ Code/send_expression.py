# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:05:20 2024

@author: 20Jan
"""


import serial
import time



arduino = serial.Serial('COM4', 9600)
time.sleep(2)  # wait for the serial connection to initialize

def send_expression_to_arduino(expression):
    command = f"{expression}\n"  # Command string to send
    arduino.write(command.encode())

# Example usage
send_expression_to_arduino('angry')
#making sure serial connection is closed properly, even if an error occurs. This lets us send more expressions by running the script again
arduino.close()
