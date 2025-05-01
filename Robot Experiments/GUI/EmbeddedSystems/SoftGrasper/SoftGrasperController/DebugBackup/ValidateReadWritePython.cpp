#include <SPI.h>



float CommandValues[7]={0,0,0,0,0,0,0};
//unsigned long HoldValues[4]={0,0,0,0};
int i = 0;
byte HoldValues[4]={0,0,0,0};

IntervalTimer PressureControlTimer;

  int incomingByte1 = 0;
  int incomingByte2 = 0;
  int incomingByte3 = 0;
  int incomingByte4 = 0;



void setup() {
  // put your setup code here, to run once:
    Serial.begin(1000000);  //apparently can communicate at 1 MBaud 
    // while (true)
    // {
    // Serial.println("Begin!");
    // Serial.println("HelloWorld!");
    // }
    //PressureControlTimer.begin(runActivationPattern, 4000);
    


}

void loop()
{
    long fval = 0;
    while (Serial.available()<=8)
    {
      continue;

    }  
    String output = "";
    //   union equiv {
    //     float x;
    //     unsigned long l;
    //   } equiv ;
    //   equiv.l = 0;

    for (int i=0;i<4;i++)
    {
      HoldValues[i]= Serial.read();
      output = output+HoldValues[i]+",";
      //equiv.l = HoldValues[i]<<int(8*(i)) | equiv.l;
      
    }
    float val = *((float*)(HoldValues));
  
    for (int i=0;i<4;i++)
    {
      HoldValues[i]= Serial.read();
      output = output+HoldValues[i]+",";
      //equiv.l = HoldValues[i]<<int(8*(i)) | equiv.l;
      
    }
    float val2 = *((float*)(HoldValues));


    Serial.println(output);
    //Serial.println(equiv.x);
    Serial.println(val);
    Serial.println(val2);
    delay(1);


    








  // if (Serial.available()==4){
  //   // String test = Serial.readStringUntil('\n');
  //   // Serial.print(test);
  //   Serial.println("Start");
  //   incomingByte1 = Serial.read();
  //   Serial.println(incomingByte1);

  //   incomingByte2 = Serial.read();
  //   Serial.println(incomingByte2);

  //   incomingByte3 = Serial.read();
  //   Serial.println(incomingByte3);

  //   incomingByte4 = Serial.read();
  //   Serial.println(incomingByte4);
  //   Serial.println("ok");


  // }

  // long val = incomingByte1<<24 + incomingByte2<<16 + incomingByte3<<8 + incomingByte4;
  // Serial.println("Val Construction");
  // Serial.print(val);
  // Serial.print("\n");
  




}



// void runActivationPattern()
// {
//     Serial.println(neuronStates1,BIN);
//     Serial.println(neuronStates2,BIN);
//     Serial.println(neuronStates3,BIN);
//     Serial.println(neuronStates4,BIN);
//     Serial.println(assembledFloat,BIN);

// }


// void read_StrCommand(){

// }


// void read_Commands(Stream &port) {
//   // reads the current neuron state from the neural controller Teensy


// }