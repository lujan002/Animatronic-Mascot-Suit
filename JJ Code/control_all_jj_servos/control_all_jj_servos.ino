#include <Servo.h>

Servo beak_l;  
Servo beak_r;  
Servo eyelid_l;
Servo eyelid_r;
Servo eyebrow_l_o;
Servo eyebrow_l_i;
Servo eyebrow_r_i;
Servo eyebrow_r_o;

void setup() {
  Serial.begin(9600);       // Start serial communication at 9600 baud
  beak_l.attach(2);         // Lower beak, left
  beak_r.attach(3);         // Lower beak, right
  eyelid_l.attach(4);       // Eyelid, left
  eyelid_r.attach(5);       // Eyelid, right
  eyebrow_l_o.attach(6);    // Eyebrow, left, outer
  eyebrow_l_i.attach(7);    // Eyebrow, left, inner
  eyebrow_r_i.attach(8);    // Eyebrow, right, inner
  eyebrow_r_o.attach(9);    // Eyebrow, right, outer
                            //(Orientation if facing mascot) 
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    if (command == "MOUTH_OPEN") {
      openBeak();
    } else if (command == "MOUTH_CLOSE") {
      closeBeak();
    if (command == "EYES_OPEN")
      openEyes();
    } else if (command == "EYS_CLOSE") {
      closeEyes();
    if (command == "SET_EMOTION_Angry") {
      setAngry();
    } else if (command == "SET_EMOTION_Happy") {
      setHappy();
    } else if (command == "SET_EMOTION_Sad") {
      setSad();
    } else if (command == "SET_EMOTION_Surprised") {
      setSurprised();
    } else if (command == "SET_EMOTION_Neutral") {
      setNeutral(); 
    }
  }
}

void openBeak() {
  servo1.write(92);  // Set servo1 to 92 degrees
  servo2.write(108); // Set servo2 to 108 degrees
}

void closeBeak() {
  servo1.write(108); // Set servo1 to 108 degrees
  servo2.write(92);  // Set servo2 to 92 degrees
}

void openEyes() {

}

void closeEyes() {

}

void setAngry() {

}

void setHappy() {

}

void setSad() {

}

void setSurprised() {

}

void setNeutral() {

}


