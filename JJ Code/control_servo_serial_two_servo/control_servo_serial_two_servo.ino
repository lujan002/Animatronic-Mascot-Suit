#include <Servo.h>

Servo servo1;
Servo servo2;
// Add more servos as needed

void setup() {
  Serial.begin(9600);
  servo1.attach(10);
  servo1.write(95);
  servo2.attach(11);
  servo2.write(95);  
  //Attach more servos as needed
}

void loop() {
  // Check if data is available to read from the serial buffer
  if (Serial.available()) {
    // Read the incoming data as a string until a newline character ('\n') is encountered
    String data = Serial.readStringUntil('\n');

    int firstCommaIndex = data.indexOf(',');

    // Extract servo values from the data string
    int servo1Value = data.substring(0, firstCommaIndex).toInt();
    int servo2Value = data.substring(firstCommaIndex + 1).toInt();

    // Send the integer values to the respective servos to set their positions
    servo1.write(servo1Value);
    servo2.write(servo2Value);
  
 
 
 
  }
}

