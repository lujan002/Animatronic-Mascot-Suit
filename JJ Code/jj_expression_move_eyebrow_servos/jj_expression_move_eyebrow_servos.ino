#include <Servo.h>

Servo servo1;
Servo servo2;

void setup() {
  Serial.begin(9600);
  servo1.attach(3);  // Attach servo1 to pin 3
  servo2.attach(5);  // Attach servo2 to pin 5
  servo1.write(90);
  servo2.write(90);
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    if (command == "normal") {
      servo1.write(94);
      servo2.write(109);
    } else if (command == "angry") {
      servo1.write(121);
      servo2.write(155);
    } else if (command == "sad") {
      servo1.write(46);
      servo2.write(113);
    } else if (command == "surprised") {
      servo1.write(100);
      servo2.write(79);
    }
    // Add a short delay to allow the servos to physically move to the position
    delay(15); // Adjust this delay as necessary
  }
}
