
#include "PressurePort.h"
#include <SPI.h>
#include <Bounce.h>



//#define useNeuralCommands //define to use pre-computed commands for feedforward
#define BITING              //behavior
#define TransferFromSerial  //real time feedback from other Teensy

//--------------debug switches---------------
// #define debug  //define to allow print statements of activations and pressures
//#define debug2 //dont print activations, only a select few pressures
#define PressureCycle  //check valves during setup
//#define DebugSingle //for debugging single port at a time
#define ReadSingle

unsigned long TimeScaleInterval = 200000;  //for static interpolate

int Ser_TScale = 100;  //for when running serial



//--------------Neural Integration-----------
// timing variables
float DT = 0.001;  // [s] delay between updates of neural controller (20 Hz goal, updated based on communication speed)

// Neuron definitions


// Integration parameters (for both activation and muscle dynamics)
float tau_I4 = Ser_TScale * 0.05 * 1.0 / sqrt(2);  //0.05 x this number
float tau_I3a = Ser_TScale * 2.0 / sqrt(2);
float tau_I3 = 1 / sqrt(2);
float tau_I2_ingestion = Ser_TScale * 0.5 / sqrt(2);
float tau_I2_egestion = Ser_TScale * 1.4 / sqrt(2);
float tau_hinge = Ser_TScale * 1 / sqrt(2);
float tau_I1 = Ser_TScale * 0.5 * 1 / sqrt(2);
float tau_opener = Ser_TScale * 1 / sqrt(2);
float tau_I3_1 = Ser_TScale * tau_I3 * 0.5;
float tau_I3_2 = Ser_TScale * 1 * tau_I3;  //originally 2* I3_1
float tau_I3_3 = Ser_TScale * 1.3 * tau_I3;
float tau_I3_4 = Ser_TScale * 1.3 * tau_I3;  //Ser_TScale*1 * tau_I3;
float tau_I3_5 = Ser_TScale * 1 * tau_I3;
float tau_I3_6 = Ser_TScale * 1 * tau_I3;




//--------------Neural Commands---------------


const int FlowRateMeterPin = A0;


//--------------Loop Timing---------------

unsigned long startTime = 0;     //uS
unsigned long endTime = 0;       //uS
unsigned long previousTime = 0;  //ms
unsigned long LoopTime = 1000;   //us
unsigned long t_init = 0;        //time when the loop starts to begin calculating sinusoid

IntervalTimer PressureControlTimer;

int count = 0;  //how many times through pressure control loop

byte bitArray[4] = { 0 };
byte MaxNumShiftRegisters = 4;
byte ArrayElementSize = 8;


//#define RCK 14
//#define G_ENABLE 15
//#define SEROUT 16

const int RCK = 14;
const int G_ENABLE = 15;





const int numPressurePorts = 12;  //Pressure ports: 1,4,5,8 and 9
PressurePort Pa[numPressurePorts] = { PressurePort(1, 2, 28, PressurePort::START, 0.4, 0, 30, "I3_1", 0.12), //actuator 0
                                      PressurePort(3, 4, 25, PressurePort::START, 0.4, 0, 30, "I3_5", 0.12),//actuator 1
                                      PressurePort(7, 8, 24, PressurePort::START, 0.4, 0, 30, "", 0.12),  //broken //actuator 2
                                      PressurePort(6, 5, 10, PressurePort::START, 0.4, 0, 30, "I3_2", 0.12), //actuator 3
                                      PressurePort(9, 10, 9, PressurePort::START, 0.4, 0, 30, "I3_3", 0.12), //actuator 4
                                      PressurePort(11, 12, 8, PressurePort::START, 0.4, 0, 30, "I3_4", 0.12), //actuator 5
                                      PressurePort(13, 14, 7, PressurePort::START, 0.4, 0, 30, "I1", 0.12), //actuator 6
                                      PressurePort(15, 16, 6, PressurePort::START, 0.4, 0, 30, "I2", 0.5), //actuator 7
                                      PressurePort(17, 18, 5, PressurePort::START, 0.4, 0, 30, "I7", 0.5), //actuator 8
                                      PressurePort(19, 20, 4, PressurePort::START, 0.4, 0, 30, "I4", 0.5), //actuator 9
                                      PressurePort(24, 23, 3, PressurePort::START, 0.4, 0, 30, "Hinge", 0.5), //actuator 10
                                      PressurePort(22, 21, 2, PressurePort::START, 0.4, 0, 30, "I3_6", 0.12) }; //actuator 11

////Kill switch for releasing pressure and stopping program
//const int buttonPin = 12;
//Bounce pushbutton = Bounce(buttonPin, 10);  // 10 ms debounce

const int FRR_Test_Port = 5;  //use act 5 to read pressure
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  SPI.begin();
  pinMode(RCK, OUTPUT);

  digitalWrite(G_ENABLE, HIGH);
  pinMode(G_ENABLE, OUTPUT);
  delay(1000);

  //pinMode(SEROUT,INPUT);

  digitalWrite(G_ENABLE, LOW);


  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));

  digitalWrite(RCK, LOW);
  unsigned long long1 = 0x00000000;
  unsigned long long2 = 0x00000000;
  long1 = SPI.transfer(0b00000000);  //(0b00100000);//relay connected to Drain 5 on 2nd Shift Register
  long2 = SPI.transfer(0b00000000);  //(0b00000100);//relay connected to Drain 2 on 1st Shift Register
  SPI.transfer(0b00000000);
  Serial.print("NEW");






  digitalWrite(RCK, HIGH);
  SPI.endTransaction();

  digitalWrite(G_ENABLE, LOW);


  //delay(1000000);
  digitalWrite(G_ENABLE, HIGH);

#ifdef PressureCycle

  for (int i = 0; i < numPressurePorts; i++) {


    //byte bitArray[4]={0};
    Pa[i].setValves(PressurePort::INFLATE);
    Pa[i].WriteToShiftRegister(Pa[i].RCK, Pa[i].G_ENABLE, Pa[i].ShiftArray, 4);
    delay(1000);

    Pa[i].setValves(PressurePort::VACUUM);
    Pa[i].WriteToShiftRegister(Pa[i].RCK, Pa[i].G_ENABLE, Pa[i].ShiftArray, 4);
    delay(1000);
  }
#endif


#ifdef DebugSingle

  while (true) {
    int muscleToTest = 11;
    Pa[muscleToTest].modulatePressure(8);
    float rawPress = Pa[muscleToTest].readPressure(false);  //without moving average
    char buffer[80];
    char pressure_s[10];
    dtostrf(rawPress, 3, 4, pressure_s);
    Serial.println(pressure_s);
    delay(1);
  }

#endif

#ifdef ReadSingle

  while (true) {
    int muscleToTest = FRR_Test_Port;
    float rawPress = Pa[muscleToTest].readPressure(false);  //without moving average
    char buffer[80];
    char pressure_s[10];
    dtostrf(rawPress, 3, 4, pressure_s);
    Serial.println(pressure_s);
    delay(1);
  }

#endif


}

void loop() {

}
