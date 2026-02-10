int soundPin = A0;   
int ledPin = 13;     
int threshold = 80; 

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600); 
}

void loop() {
  int sensorValue = analogRead(soundPin); 
  
  
  Serial.print("Sound Level: ");
  Serial.println(sensorValue);

  if (sensorValue > threshold) {
    digitalWrite(ledPin, HIGH); 
    delay(2000);                 
  } else {
    digitalWrite(ledPin, LOW);
  }
}