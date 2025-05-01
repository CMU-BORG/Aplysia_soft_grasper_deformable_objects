#include "PressurePort.h"
#include <SPI.h>
#include <Bounce.h>
#include <Wire.h>




//--------------debug switches---------------


enum PortAction_ENUM
{
    IGNORE = 0,
    READ_FORCE = 1,
};



struct PortActions
{
    PortAction_ENUM action = READ_FORCE;
    int analogPin = 18;
    float readADC = 0.0; //read pressure

};

struct PortActions PortA = { READ_FORCE,18,0.0};


//-------------- Load Cell ---------------//
union ByteToInt {
  byte arr_v[2];
  uint32_t int_v;
};

//-------------- Function definitions ---------------//
void readForce();


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Wire.begin();

  
  delay(1000);

  String command = "C1";

  while (true) {

    

      
      // read the incoming byte:
      
      delay(10);
      if (Serial.available() > 0) 
      { 
        command = Serial.readStringUntil('\r').trim();
      }
      
      if (command == "C1")
        {
          readForce();
          float rawForce = PortA.readADC;  //without moving average

          char buffer[80];
          char force_s[15];
          dtostrf(rawForce, 3, 4, force_s);
          Serial.print(force_s);
          Serial.print("\r");
          delay(1);
  
        }

    }

    
      


}


void loop() {

}

void readForce(){

  ByteToInt sensorB;
  byte num[2]={0,0};
  byte mask[2] = {B00111111,B11111111}; // Binary mask.  will receive two bytes, the 1st byte is the MSB, the 2nd byte is the LSB.  Only keep the LS 6 bits for the 1st byte



  // read 1 byte, from address 0
  Wire.requestFrom(40, 2);
  int inc = 0;
  while(Wire.available()) {
    byte num_v = Wire.read();
    num[inc] = num_v & mask[inc];
    //Serial.print(num[inc], BIN);
    //Serial.print(",");
    inc++; // reverse bit order
  }
  //Serial.println(inc);

  // Reverse Bit Order
  for (int j = 0;j<2;j++)
  {
    sensorB.arr_v[j] = num[1-j];
  }

  //Serial.println(sensorB.int_v, HEX);
  //Serial.println(sensorB.int_v, DEC);

  PortA.readADC = sensorB.int_v/1.0;  //save value as a float.




}
