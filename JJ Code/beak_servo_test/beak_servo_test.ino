#include <Servo.h>

// initialize all 4 servos
Servo servo1;


int servo_1_angle_start = 60;
int servo_1_angle_end =160;


void setup() {
  // put your setup code here, to run once:
  servo1.attach(3);

}

void loop (){
  servo1.write(160);
  delay(500);
  servo1.write(60);
  delay(500);
}
