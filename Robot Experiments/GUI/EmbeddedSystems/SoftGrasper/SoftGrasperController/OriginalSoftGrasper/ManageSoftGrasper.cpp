#include "PressurePort.h"
#include <SPI.h>
#include <Bounce.h>


const int RCK = 14;
const int G_ENABLE = 15;


//#define PressureCycle
#define debug
#define DeflateALL
//#define ReadSingle
//#define DebugSingle
//#define ReadMultiple


// ------------- Initial Pressure Controller ---------------- //

// Controller arguments:
// int NumValveInflation, int NumValveVacuum, int pressureSensor_SS, PressureState state, float pressure_tolerance, float min_pressure, float max_pressure, String Muscle, float alpha
const int numPressurePorts = 12;  
const int numActivePorts = 7;
PressurePort Pa[numPressurePorts] = { PressurePort(1, 2, 28, PressurePort::START, 0.1, 0, 30, "I3", 0.01), //actuator 0 h
                                      PressurePort(3, 4, 25, PressurePort::START, 0.4, 0, 30, "Not Used", 0.12),//actuator 1 h
                                      PressurePort(7, 8, 24, PressurePort::START, 0.4, 0, 30, "Not Used", 0.12),  //broken //actuator 2
                                      PressurePort(6, 5, 10, PressurePort::START, 0.1, 0, 30, "I1_1", 0.12), //actuator 3 h
                                      PressurePort(9, 10, 9, PressurePort::START, 0.1, 0, 30, "I1_2", 0.12), //actuator 4 h
                                      PressurePort(11, 12, 8, PressurePort::START, 0.4, 0, 30, "Not Used", 0.5), //actuator 5 h
                                      PressurePort(13, 14, 7, PressurePort::START, 0.1, 0, 30, "I1_3", 0.12), //actuator 6 h
                                      PressurePort(15, 16, 6, PressurePort::START, 0.05, 0, 30, "Not Used", 0.4), //actuator 7 l
                                      PressurePort(17, 18, 5, PressurePort::START, 0.05, 0, 30, "JAW1", 0.35), //actuator 8 l 
                                      PressurePort(19, 20, 4, PressurePort::START, 0.05, 0, 30, "JAW2", 0.35), //actuator 9 l 
                                      PressurePort(24, 23, 3, PressurePort::START, 0.05, 0, 30, "Not Used", 0.5), //broken //actuator 10 h
                                      PressurePort(22, 21, 2, PressurePort::START, 0.05, 0, 30, "JAW3", 0.35) }; //actuator 11 l


//--------------Loop Timing--------------- //

unsigned long startTime = 0;     //uS
unsigned long endTime = 0;       //uS
unsigned long previousTime = 0;  //ms
unsigned long LoopTime = 1000;   //us
unsigned long t_init = 0;        //time when the loop starts to begin calculating sinusoid


void runActivationPattern();
void read_Commands();

IntervalTimer PressureControlTimer;

int count = 0;  //how many times through pressure control loop

// -------------- Shift Register Array for writing to the pressure controller valves ---------- //
byte bitArray[4] = { 0 };
byte MaxNumShiftRegisters = 4;
byte ArrayElementSize = 8;

// -------------  Pressure values for the muscle -----------------// 
float CommandValues[7]={0,0,0,0,0,0,0};
float P_I3=0;           // pressure for the I3-like circumferential muscle
float P_Jaw1=1;         // pressure for Soft Jaw 1
float P_Jaw2=1;         // pressure for Soft Jaw 2
float P_Jaw3=1;         // pressure for Soft Jaw 3
float P_I1_1=0;         // pressure for I1-like muscle for tilt control 
float P_I1_2=0;         // pressure for I1-like muscle for tilt control 
float P_I1_3=0;         // pressure for I1-like muscle for tilt control 

long HoldOverride = 0;  // read over serial if should override the pressure modulation function
// int JawCommand1 = false;
// int JawCommand2 = false;
// int JawCommand3 = false;
byte HoldValues[4]={0,0,0,0};


void setup() {
  // put your setup code here, to run once:
    Serial.begin(115200);  //apparently can communicate at 1 MBaud 
    SPI.begin();
    pinMode(RCK, OUTPUT);

    digitalWrite(G_ENABLE, HIGH);
    pinMode(G_ENABLE, OUTPUT);
    delay(1000);

    //pinMode(SEROUT,INPUT);

    digitalWrite(G_ENABLE, LOW);


    SPI.beginTransaction(SPISettings(115200, MSBFIRST, SPI_MODE0));

    digitalWrite(RCK, LOW);
    unsigned long long1 = 0x00000000;
    unsigned long long2 = 0x00000000;
    long1 = SPI.transfer(0b00000000);  //(0b00100000);//relay connected to Drain 5 on 2nd Shift Register
    long2 = SPI.transfer(0b00000000);  //(0b00000100);//relay connected to Drain 2 on 1st Shift Register
    SPI.transfer(0b00000000);







    digitalWrite(RCK, HIGH);
    SPI.endTransaction();

    digitalWrite(G_ENABLE, LOW);


    //delay(1000000);
    digitalWrite(G_ENABLE, HIGH);


    #ifdef DeflateALL
        for (int i = 0; i < numPressurePorts; i++)
        {

        Pa[i].setValve_state(PressurePort::VACUUM,bitArray,MaxNumShiftRegisters, ArrayElementSize);  //keep valves closed
        
       
        }
    #endif
    PressurePort::WriteToShiftRegister(PressurePort::RCK, PressurePort::G_ENABLE, bitArray, MaxNumShiftRegisters); //executes the bitArray to the shift registers to actually open or close the valves 
    delay(8000);





    // -- Pressure Cycle -- //
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

     // -- Debug A single Pressure Port -- //
    #ifdef DebugSingle

        int i = 0;
        int muscleToTest = 0;

        while (1) {
            Pa[muscleToTest].modulatePressure(12);
            float rawPress = Pa[muscleToTest].readPressure(false);  //without moving average
            char buffer[80];
            char pressure_s[10];
            dtostrf(rawPress, 3, 4, pressure_s);
            Serial.println(pressure_s);
            delay(1);

        }



    #endif





    // -- Debug A single Pressure Port that stops after 20 seconds -- //
    #ifdef DebugSingleStop

        int i = 0;
        int muscleToTest = 0;

        while (i<20000) {
            Pa[muscleToTest].modulatePressure(12);
            float rawPress = Pa[muscleToTest].readPressure(false);  //with moving average
            char buffer[80];
            char pressure_s[10];
            dtostrf(rawPress, 3, 4, pressure_s);
            Serial.println(pressure_s);
            delay(1);
            i++;
        }
        while (1)
        {
            Pa[muscleToTest].modulatePressure(0);
        }


    #endif


    // -- Read from a single pressure port -- //
    #ifdef ReadSingle
        const int FRR_Test_Port = 8;  //use act 6 to read pressure
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

    #ifdef ReadMultiple
        
        const int FRR_Test_Port [12] = {0,1,2,3,4,5,6,7,8,9,10,11};  //Bank of 4 low pressure
            while (true) {
                String outputs ="";
                for (int j=0;j<12;j++)
                {
                    int muscleToTest = FRR_Test_Port[j];
                    float rawPress = Pa[muscleToTest].readPressure(false);  //without moving average
                    char buffer[80];
                    char pressure_s[10];
                    dtostrf(rawPress, 3, 4, pressure_s);
                    dtostrf(rawPress, 3, 4, pressure_s);
                    outputs = outputs + "," + pressure_s ;
                }
                Serial.println(outputs);
                delay(1);
            }

    #endif


    #ifdef GOcommand
        //wait for Go command
        bool exit = false;
        
        while(exit!=true)
        {
            

            if (Serial.available()) 
            {
                delayMicroseconds(1000);

                String test = Serial.readStringUntil('\n');
                if (test == "GO")
                {
                    exit = true;
                }
                delayMicroseconds(100);
                Serial.println("RD"); // send ready command
            }

        }
        Serial.println("ReadyToGo");

    #endif

    // Cycle through Jaws until pressure is at setpoint
    bool KeepLooping=true;
    int JawNumber[3] = {8,9,11};
    float JawPressure[3] ={0,0,0};
    float JawPressureSetpoint[3] = {1,1,1};
    float JawPressureTolerance[3]={0.05,0.05,0.05};
    bool lockJawValves[3]={0,0,0};
    
    String waitf = Serial.readStringUntil('\n');
    Serial.println("Begin Jaw Inflate");

    while(KeepLooping)
    {
        String outputs="";
        for (int i = 0; i < numPressurePorts; i++)
        {

            Pa[i].setValve_state(PressurePort::HOLD,bitArray,MaxNumShiftRegisters, ArrayElementSize);  //keep valves closed
            float rawPress = Pa[i].readPressure(false);  //without moving average
            char buffer[80];
            char pressure_s[10];
            dtostrf(rawPress, 3, 4, pressure_s);
            dtostrf(rawPress, 3, 4, pressure_s);
            outputs = outputs + "," + pressure_s ;
            
        }
        Serial.println(outputs);

        for(int i=0;i<3;i++)
        {
            JawPressure[i] = Pa[JawNumber[i]].readPressure(false);
            
            float JawPressErr = abs(JawPressure[i]-JawPressureSetpoint[i]);
            if (JawPressErr>=JawPressureTolerance[i] & lockJawValves[i]==0)
            {
                float activation = JawPressureSetpoint[i]/30;
                Serial.println(activation);
                Pa[JawNumber[i]].modulatePressure_Activation(activation, bitArray, MaxNumShiftRegisters, ArrayElementSize  );  //writes to bitArray which valves to open or close

                
            }
            else
            {
                lockJawValves[i] = 1;

            }

        }

        if (lockJawValves[0]==1 & lockJawValves[1]==1 & lockJawValves[2]==1)
        {
            KeepLooping=false;

        }        

        PressurePort::WriteToShiftRegister(PressurePort::RCK, PressurePort::G_ENABLE, bitArray, MaxNumShiftRegisters); //executes the bitArray to the shift registers to actually open or close the valves 
        
       




    }

     Serial.println("Jaws finished!");



    // -- Run the interval timer -- //
    //PressureControlTimer.begin(runActivationPattern, 1000);


}

void loop()
{
    //Serial.println(endTime-startTime);
    long fval = 0;
    while (Serial.available()<=4)
    {

      continue;

    }  

    for (int i=0;i<4;i++)
    {
      HoldValues[i]= Serial.read();
      
    }
    float val = *((float*)(HoldValues));
    CommandValues[3] = val;
    runActivationPattern();
    delayMicroseconds(100);
    

    
     



}



void runActivationPattern()
{
    //get commanded pressure values
    startTime = micros();
    //read_Commands();
    // Serial.println("DL");
    // delayMicroseconds(100);
    //  for (int v =0;v<numActivePorts;v++)
    // {
    //     Serial.println(CommandValues[v]);
    // }

    //--------------Timing begin
    unsigned long currentTime = micros();
    

    if (count == 0)
    {
        t_init = currentTime;
        count++;
        //P1.setValves(PressurePort::INFLATE);
    }

    unsigned long t = currentTime - t_init;
    float T = 1000000;

    //-----------------------------

    float activation[numPressurePorts] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    float highflowPress = 12;
    float lowflowPress = 8;

    P_Jaw1 = CommandValues[0];
    P_Jaw2 = CommandValues[1];
    P_Jaw3 = CommandValues[2];
    P_I3 = CommandValues[3];
    P_I1_1 = CommandValues[4];
    P_I1_2 = CommandValues[5];
    P_I1_3 = CommandValues[6];


    activation[0] = P_I3/30;
    activation[8] = P_Jaw1/30;
    activation[9] = P_Jaw2/30;
    activation[11] = P_Jaw3/30;
    activation[3] = P_I1_1/30;
    activation[4] = P_I1_2/30;
    activation[6] = P_I1_3/30;


    
    String outputs = "";
    for (int i = 0; i < numPressurePorts; i++)
    {

        Pa[i].setValve_state(PressurePort::HOLD,bitArray,MaxNumShiftRegisters, ArrayElementSize);  //keep valves closed
        float rawPress = Pa[i].readPressure(false);  //without moving average
        char buffer[80];
        char pressure_s[10];
        dtostrf(rawPress, 3, 4, pressure_s);
        dtostrf(rawPress, 3, 4, pressure_s);
        outputs = outputs + "," + pressure_s ;
        
       
    }

    Pa[0].modulatePressure_Activation(activation[0], bitArray, MaxNumShiftRegisters, ArrayElementSize  );  //writes to bitArray which valves to open or close
    Serial.println(outputs);







    PressurePort::WriteToShiftRegister(PressurePort::RCK, PressurePort::G_ENABLE, bitArray, MaxNumShiftRegisters); //executes the bitArray to the shift registers to actually open or close the valves 
    endTime = micros();
    
}








void read_Commands() 
{
    while (Serial.available()<=4)
    {
        continue;
    }
    byte HoldValues[4] = {0,0,0,0};
    for (int i=0;i<4;i++)
    {
        HoldValues[i]= Serial.read();
                    
    }

    float val = *((float*)(HoldValues));
    CommandValues[3] = val;
}

void read_Commands2() {
    while (Serial.available()<=28)
    {
        Serial.println(Serial.available());
    continue;
    }



    for (int j=0;j<7;j++)
    {
        byte HoldValues[4] = {0,0,0,0};
        for (int i=0;i<4;i++)
        {
            HoldValues[i]= Serial.read();
                        
        }

        CommandValues[j] = *((float*)(HoldValues));
    }


    // Get Pressure readings
    String outputs = "";
    for (int i=0;i<12;i++)
    {
        float rawPress = Pa[i].readPressure(false); //without moving average
        char buffer[80];
        char pressure_s[10];
        dtostrf(rawPress, 3, 4, pressure_s);
        outputs = outputs + pressure_s +"," ;
        
    }
    Serial.println(outputs);
    
Serial.println("ok");


  
}