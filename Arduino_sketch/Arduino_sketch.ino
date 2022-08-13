void setup() {
  pinMode(13, OUTPUT);
  Serial.begin(9600);
  digitalWrite(13, LOW);
}

byte input = 0;
void loop() {
  if (Serial.available() > 0) {
    input = (byte)Serial.read();
    if(input == 1){
      digitalWrite(13, HIGH);
    }else{
      digitalWrite(13, LOW);
    }
  }
}
