#include <Servo.h>

// initialize all 4 servos
Servo servo1;
Servo servo2;

int servo_1_angle_start = 55;
int servo_1_angle_end = 125;
int servo_2_angle_start = 0;
int servo_2_angle_end = 45;


void setup() {
  // put your setup code here, to run once:
  servo1.attach(8);
  servo2.attach(9);
  servo2.write(0);
}

void loop (){
  // put your main code here, to run repeatedly:
  for (int i1 = servo_1_angle_start; i1 < servo_1_angle_end; i1++) {
    servo1.write(i1);
    delay(5);
  }
  for (int i1 = servo_1_angle_end; i1 > servo_1_angle_start; i1--) {
    servo1.write(i1);
    delay(5);
  }
  
  for (int i2 = servo_2_angle_start; i2 < servo_2_angle_end; i2++) {
    servo2.write(i2);
    delay(5);
  }
  for (int i2 = servo_2_angle_end; i2 > servo_2_angle_start; i2--) {
    servo2.write(i2);
    delay(5);
  }
}

