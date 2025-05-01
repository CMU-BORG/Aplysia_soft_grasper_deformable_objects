#include "PressurePort.h"
#include <SPI.h>
#include <Bounce.h>




//--------------debug switches---------------
// #define debug  //define to allow print statements of activations and pressures
//#define debug2 //dont print activations, only a select few pressures
#define PressureCycle  //check valves during setup
//#define DebugSingle //for debugging single port at a time
#define ReadSingle


//--------------Loop Timing---------------

unsigned long startTime = 0;     //uS
unsigned long endTime = 0;       //uS
unsigned long previousTime = 0;  //ms
unsigned long LoopTime = 1000;   //us
unsigned long t_init = 0;        //time when the loop starts to begin calculating sinusoid



IntervalTimer PressureControlTimer;

int count = 0;  //how many times through pressure control loop


//#define RCK 14
//#define G_ENABLE 15
//#define SEROUT 16

const int RCK = 14;
const int G_ENABLE = 15;





const int numPressurePorts = 2;  //Pressure ports: 1,4,5,8 and 9
PressurePort Pa[numPressurePorts] = { PressurePort(1, 2, 1, PressurePort::START, 0.4, 0, 30, "I3_1", 0.12), //actuator 0
                                      PressurePort(3, 4, 28, PressurePort::START, 0.4, 0, 30, "I3_5", 0.12)//actuator 1
                                      }; //actuator 11

////Kill switch for releasing pressure and stopping program
//const int buttonPin = 12;
//Bounce pushbutton = Bounce(buttonPin, 10);  // 10 ms debounce

const int FRR_Test_Port[2] = {0,1};  //use act 5 to read pressure

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  SPI.begin();
  
  delay(1000);


  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));
  SPI.endTransaction();




#ifdef ReadSingle
String command = "C1";
  while (true) {

      
      // read the incoming byte:
      
      delay(10);
      if (Serial.available() > 0) 
      { 
        command = Serial.readStringUntil('\r').trim();
        //Serial.print(command);
      }
      

      if (command == "C")
      {
        for (int i=0;i<numPressurePorts;i++)
        {
          int muscleToTest = FRR_Test_Port[i];
          float rawPress = Pa[muscleToTest].readPressure(false);  //without moving average

          char buffer[80];
          char pressure_s[10];
          dtostrf(rawPress, 3, 4, pressure_s);
          Serial.print(pressure_s);


          if (i== numPressurePorts-1)
          {
            Serial.print("\r");
          }

          else
          {
            Serial.print(",");
          }
          }

      }

      if (command == "C1")
        {

          int muscleToTest = FRR_Test_Port[0];
          float rawPress = Pa[muscleToTest].readPressure(false);  //without moving average

          char buffer[80];
          char pressure_s[10];
          dtostrf(rawPress, 3, 4, pressure_s);
          Serial.print(pressure_s);
          Serial.print("\r");
          delay(1);
  
        }

      if (command == "C2")
        {
            int muscleToTest = FRR_Test_Port[1];
            float rawPress = Pa[muscleToTest].readPressure(false);  //without moving average
  
            char buffer[80];
            char pressure_s[10];
            dtostrf(rawPress, 3, 4, pressure_s);
            Serial.print(pressure_s);
            Serial.print("\r");
  
        }

    }

  
    

#endif


}


void loop() {

}
