const int limitSwitchPin = 2; // Pin connected to the limit switch

volatile int switchState = HIGH; // volatile keyword is used as this variable changes inside an interrupt

void setup() {
  Serial.begin(9600);
  pinMode(limitSwitchPin, INPUT_PULLUP); // Use internal pull-up resistor
  attachInterrupt(digitalPinToInterrupt(limitSwitchPin), toggleSwitch, CHANGE); // Attach interrupt
}

void loop() {
  if (switchState == LOW) {
    Serial.println("Low"); // Pressed
  } else {
    Serial.println("High"); // Not pressed
  }
  delay(100); // Delay to reduce serial output speed
}

void toggleSwitch() {
  switchState = digitalRead(limitSwitchPin); // Read the new state of the limit switch
}
