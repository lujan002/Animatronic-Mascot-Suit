#import libraries
import RPi.GPIO as GPIO
import time

#GPIO Basic initialization
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

#Use a variable for the Pin to use
Angry = 4
Happy = 5
Sad = 6
Surprised = 7
Neutral = 8


#Initialize your pin
GPIO.setup(Angry,GPIO.OUT)
GPIO.setup(Happy,GPIO.OUT)
GPIO.setup(Sad,GPIO.OUT)
GPIO.setup(Surprised,GPIO.OUT)
GPIO.setup(Neutral,GPIO.OUT)

#Turn on the LED
print("LED on")
GPIO.output(Angry,1)
GPIO.output(Happy,1)
GPIO.output(Sad,1)
GPIO.output(Surprised,1)
GPIO.output(Neutral,1)

#Wait 5s
time.sleep(5)

#Turn off the LED
print("LED off")
GPIO.output(Angry,0)
GPIO.output(Happy,0)
GPIO.output(Sad,0)
GPIO.output(Surprised,0)
GPIO.output(Neutral,0)

