int ldrPin = 2; 

void setup() {
  pinMode(ldrPin, INPUT);
  Serial.begin(9600);
  Serial.println("LDR Digital Test Starting...");
}

void loop() {
  int status = digitalRead(ldrPin); 

  if (status == HIGH) {
    Serial.println("Status: DARK");
  } else {
    Serial.println("Status: LIGHT");
  }
  
  delay(200);
}