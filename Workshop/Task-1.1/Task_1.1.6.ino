void setup() {
  Serial.begin(9600);
}

void loop() {
  int flameVal = analogRead(A0); 
  Serial.print("Flame Sensor Value: ");
  Serial.println(flameVal); 
  delay(100);
}
