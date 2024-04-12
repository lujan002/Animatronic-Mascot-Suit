#include <Servo.h>

Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
// Add more servos as needed

void setup() {
  Serial.begin(9600);
  servo1.attach(3);
  servo1.write(40);
  servo2.attach(5);
  servo2.write(90);  
  servo3.attach(6);
  servo3.write(90);
  servo4.attach(9);
  servo4.write(90);
  //Attach more servos as needed
}

void loop() {
  // Check if data is available to read from the serial buffer
  if (Serial.available()) {
    // Read the incoming data as a string until a newline character ('\n') is encountered
    String data = Serial.readStringUntil('\n');

    int firstCommaIndex = data.indexOf(',');
    int secondCommaIndex = data.indexOf(',', firstCommaIndex + 1);
    int thirdCommaIndex = data.indexOf(',', secondCommaIndex + 1);

    // Extract servo values from the data string
    int servo1Value = data.substring(0, firstCommaIndex).toInt();
    int servo2Value = data.substring(firstCommaIndex + 1, secondCommaIndex).toInt();
    int servo3Value = data.substring(secondCommaIndex + 1, thirdCommaIndex).toInt();
    int servo4Value = data.substring(thirdCommaIndex + 1).toInt(); // To the end of the string
    
    // Send the integer values to the respective servos to set their positions
    servo1.write(servo1Value);
    servo2.write(servo2Value);
    servo3.write(servo3Value);
    servo4.write(servo4Value);
  }
}

