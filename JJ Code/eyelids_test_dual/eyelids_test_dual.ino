#include <Servo.h>

// Define the pin numbers for the limit switches for the first eyelid
const int limitSwitchPin1 = 2;
const int limitSwitchPin2 = 3;
// Define the pin numbers for the limit switches for the second eyelid
const int limitSwitchPin3 = 4;
const int limitSwitchPin4 = 5;

// Define the pin numbers for the servos
const int servoPin1 = 10;
const int servoPin2 = 11;

Servo myServo1;  // Create a servo object for the first eyelid
Servo myServo2;  // Create a servo object for the second eyelid

void setup() {
  // Initialize the Serial Monitor at 9600 bps
  Serial.begin(9600);
  
  // Configure the limit switch pins as input with internal pull-up resistor enabled
  pinMode(limitSwitchPin1, INPUT_PULLUP);
  pinMode(limitSwitchPin2, INPUT_PULLUP);
  pinMode(limitSwitchPin3, INPUT_PULLUP);
  pinMode(limitSwitchPin4, INPUT_PULLUP);

  // Attach the servos to the respective pins
  myServo1.attach(servoPin1);
  myServo2.attach(servoPin2);

  // Move servos to initial position (half speed in one direction)
  myServo1.write(0); // Adjust this value for half-speed
  myServo2.write(0); // Adjust this value for half-speed
}

void loop() {
  // Read the state of the limit switches for the first eyelid
  int stateSwitch1 = digitalRead(limitSwitchPin1);
  int stateSwitch2 = digitalRead(limitSwitchPin2);
  // Read the state of the limit switches for the second eyelid
  int stateSwitch3 = digitalRead(limitSwitchPin3);
  int stateSwitch4 = digitalRead(limitSwitchPin4);
  
  // Control the first servo based on the limit switch states
  if (stateSwitch1 == LOW) { // If switch 1 is pressed
    // Reverse direction to half speed other way
    myServo1.write(180); // Adjust this value for half-speed in the reverse direction
    myServo2.write(180); // Synchronize the second servo
  } else if (stateSwitch2 == LOW) { // If switch 2 is pressed
    // Reverse direction to half speed one way
    myServo1.write(0); // Adjust this value for half-speed in the original direction
    myServo2.write(0); // Synchronize the second servo
  }
  
  // Similarly, control the second servo if needed separately (optional, remove if not needed)
  if (stateSwitch3 == LOW) {
    myServo2.write(180);
  } else if (stateSwitch4 == LOW) {
    myServo2.write(0);
  }

  // Add a delay to prevent bouncing effects from the switch
  delay(10); // Delay for 10 milliseconds
}
