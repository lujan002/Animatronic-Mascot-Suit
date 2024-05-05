#include <Servo.h>

// Define the pin numbers for the limit switches
const int limitSwitchPin1 = 3;
const int limitSwitchPin2 = 2;
// Define the pin number for the servo
const int servoPin = 10;

Servo myServo;  // Create a servo object

void setup() {
  // Initialize the Serial Monitor at 9600 bps
  Serial.begin(9600);
  
  // Configure the limit switch pins as input with internal pull-up resistor enabled
  pinMode(limitSwitchPin1, INPUT_PULLUP);
  pinMode(limitSwitchPin2, INPUT_PULLUP);

  // Attach the servo on pin x to the servo object
  myServo.attach(servoPin);

  // Move servo to initial position (half speed in one direction)
  myServo.write(0); // Adjust this value for half-speed
}

void loop() {
  // Read the state of the limit switches
  int stateSwitch1 = digitalRead(limitSwitchPin1);
  int stateSwitch2 = digitalRead(limitSwitchPin2);
  
  // Control the servo based on the limit switch states
  if (stateSwitch1 == LOW) { // If switch 1 is pressed
    // Reverse direction to half speed other way
    myServo.write(180); // Adjust this value for half-speed in the reverse direction
  } else if (stateSwitch2 == LOW) { // If switch 2 is pressed
    // Reverse direction to half speed one way
    myServo.write(0); // Adjust this value for half-speed in the original direction
  }

  // Add a delay to prevent bouncing effects from the switch
  delay(10); // Delay for 10 milliseconds
}
