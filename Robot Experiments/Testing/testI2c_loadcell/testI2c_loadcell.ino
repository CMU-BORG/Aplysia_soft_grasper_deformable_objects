#include <Wire.h>

union ByteToInt {
  byte arr_v[2];
  uint32_t int_v;
};


void setup() 
{
  Wire.begin();
  Serial.begin(9600);
  // pinMode(PIN_D18, INPUT_PULLUP);
  // pinMode(PIN_D19, INPUT_PULLUP);
  // digitalWrite(PIN_D18, HIGH);
  // digitalWrite(PIN_D19, HIGH); 
  Serial.println("Connected");
}

void loop() 
{
  Serial.println("InLoop"); //https://forum.arduino.cc/t/convert-byte-array-to-uint32_t/116783/5
  ByteToInt sensorB;
  byte num[2]={0,0};
  byte mask[2] = {B00111111,B11111111}; // Binary mask.  will receive two bytes, the 1st byte is the MSB, the 2nd byte is the LSB.  Only keep the LS 6 bits for the 1st byte
  
  // set the 24C256 eeprom address to 0
  // Wire.beginTransmission(80);
  // Wire.write(0);  // address high byte
  // Wire.write(0);  // address low byte
  // Wire.endTransmission();
  
  // read 1 byte, from address 0
  Wire.requestFrom(40, 2);
  int inc = 0;
  while(Wire.available()) {
    byte num_v = Wire.read();
    num[inc] = num_v & mask[inc];
    Serial.print(num[inc], BIN);
    Serial.print(",");
    inc++; // reverse bit order
  }
  Serial.println(inc);

  // Reverse Bit Order
  for (int j = 0;j<2;j++)
  {
    sensorB.arr_v[j] = num[1-j];
  }

  Serial.println(sensorB.int_v, HEX);
  Serial.println(sensorB.int_v, DEC);


  //Serial.print("num = ");
  //Serial.println(num, DEC);
  Serial.println("EndLoop");

  delay(100);
}