#include "PressurePort.h"
#include <SPI.h>
#include <Bounce.h>


const int RCK = 14;
const int G_ENABLE = 15;


#define PressureCycle //define if you want to do the pressure cycle
//#define debug
#define DeflateALL
//#define ReadSingle
//#define DebugSingle
//#define ReadMultiple
#define DebugPrint //define if you want to send all the debug print statements




// ------------- Initial Pressure Controller ---------------- //

// Controller arguments:
// int NumValveInflation, int NumValveVacuum, int pressureSensor_SS, PressureState state, float pressure_tolerance, float min_pressure, float max_pressure, String Muscle, float alpha
const int numPressurePorts = 4;  
const int numActivePorts = 4;
PressurePort Pa[numPressurePorts] = { PressurePort(1, 2, 28, PressurePort::START, 0.1, 0, 30, "I3", 0.01), //actuator 0 h
                                      PressurePort(3, 4, 4, PressurePort::START, 0.05, 0, 30, "JAW1", 0.35), //actuator 8 l
                                      PressurePort(5, 6, 3, PressurePort::START, 0.05, 0, 30, "JAW2", 0.35), //actuator 8 l 
                                      PressurePort(7, 8, 2, PressurePort::START, 0.05, 0, 30, "JAW3", 0.35)}; //actuator 11 l


//--------------Loop Timing--------------- //

unsigned long startTime = 0;     //uS
unsigned long endTime = 0;       //uS
unsigned long previousTime = 0;  //ms
unsigned long LoopTime = 1000;   //us
unsigned long t_init = 0;        //time when the loop starts to begin calculating sinusoid

IntervalTimer PressureControlTimer;

int count = 0;  //how many times through pressure control loop

// -------------- Shift Register Array for writing to the pressure controller valves ---------- //
byte bitArray[4] = { 0 };
byte MaxNumShiftRegisters = 4;
byte ArrayElementSize = 8;


//-------------  Pressure Value -------------//
// rxBuffer: https://forum.arduino.cc/t/proper-method-of-receiving-uart-packet-data-far-exceeding-rx-buffer-size/611982/8



//-------------  Reading from Python -------------//
//Message Structure Lengths
#define MSG_HDR_LEN 2
#define MSG_TRLR_LEN 2
#define MSG_PAYLOAD_LEN_MAX 36 

//state names
#define ST_Header 0
#define ST_Payload 1
#define ST_Trailer 2

const char HeaderStr[MSG_HDR_LEN+1] = {'>', '!'};
const char TrailerStr[MSG_TRLR_LEN+1] = {'!', '<'};
const int numExpectedBytes = numPressurePorts/2;

byte Payload[MSG_PAYLOAD_LEN_MAX];

String PressureCommand;
String PortCommand;



enum PortAction_ENUM
{
    HOLD = 0,
    INFLATE = 1,
    VACUUM = 2,
    START = 3,
    OPEN = 4,
    INFLATE_AND_STOP = 5,
    INFLATE_AND_MODULATE = 6,
    IGNORE = 7,

};


struct PortActions
{
    int port = 0;
    PortAction_ENUM action = IGNORE;
    float pressure = 0.0; //commanded pressure
    float readPressure_psi = 0.0; //read pressure
};

struct PortActions PortA[numPressurePorts] = { {0, IGNORE, 0.0, 0.0},
                                            {1, IGNORE, 0.0, 0.0},
                                            {2, IGNORE, 0.0, 0.0},
                                            {3, IGNORE, 0.0, 0.0}}; 


//-------Function defs---------:
void runActivationPattern();
void read_Commands();
void ReadSerial();
void executePressurePortActions();
void TransmitPayload(char startChar[], byte protocolType,byte numBytes,  byte payload[], char endChar[] );
void TransmitPressureReadings();

void setup() {
  // put your setup code here, to run once:
    Serial.begin(460800);  //apparently can communicate at 1 MBaud 
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
        
        const int FRR_Test_Port [numPressurePorts] = {0,1,2,3};  //Bank of 4 low pressure
            while (true) {
                String outputs ="";
                for (int j=0;j<numPressurePorts;j++)
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

    char initstr[50] = "";
    strcpy(initstr,"Teensy Ready!");
    TransmitPayload(HeaderStr,1,strlen(initstr),initstr,TrailerStr);
    
    delay(10);


}

void loop()
{
    // //Serial.println(endTime-startTime);
    // long fval = 0;
    // while (Serial.available()<=4)
    // {

    //   continue;

    // }  

    // for (int i=0;i<4;i++)
    // {
    //   HoldValues[i]= Serial.read();
      
    // }
    // float val = *((float*)(HoldValues));
    // CommandValues[3] = val;
    // runActivationPattern();
    ReadSerial();
    executePressurePortActions();

    
    //delayMicroseconds(1000);

}

void ReadSerial()
{ //inspired by: https://forum.arduino.cc/t/proper-method-of-receiving-uart-packet-data-far-exceeding-rx-buffer-size/611982/8


    byte cByte;

    static int nIdx = 0;
    static int numPressure = 0;
    static int PressVal_ind [numPressurePorts] = {0,0,0,0};

    static byte stateRx = ST_Header; 

    #ifdef DebugPrint

        char debugstr[100] = "Header";

    #endif

    if(Serial.available()>0)
    {
        while(Serial.available())
        {
            cByte=Serial.read();


            switch( stateRx)
            {
                case ST_Header:

                    #ifdef DebugPrint
                        strcpy(debugstr,"Header");
                        TransmitPayload(HeaderStr,1,strlen(debugstr),debugstr,TrailerStr);
                    #endif

                    if(cByte == HeaderStr[nIdx])
                    {
                        nIdx++;

                        if(nIdx == MSG_HDR_LEN) //reached end of the handler length
                        {
                            stateRx = ST_Payload;
                            nIdx = 0; //reset idx to 0 for use in the ST_Payload
                        }
                    }

                    else
                    {
                        nIdx = 0;
                    }

                break;

                case ST_Payload: //check Payload

                    Payload[nIdx] = cByte;

                    #ifdef DebugPrint
                        strcpy(debugstr,"In Payload");
                        TransmitPayload(HeaderStr,1,strlen(debugstr),debugstr,TrailerStr);
                    #endif

                    

                    if (nIdx<numExpectedBytes)
                    {

                        PortA[2*nIdx].action = (int) (Payload[nIdx] & 0b00001111 );
                        if (PortA[2*nIdx].action == INFLATE_AND_MODULATE or PortA[2*nIdx].action == INFLATE_AND_STOP)
                        {
                            PressVal_ind[numPressure] = 2*nIdx; 
                            numPressure++; //increment how many pressure bytes to expect
                        }

                        PortA[2*nIdx+1].action = (int) (Payload[nIdx] >>4);
                        if (PortA[2*nIdx+1].action == INFLATE_AND_MODULATE or PortA[2*nIdx+1].action == INFLATE_AND_STOP)
                        {
                            PressVal_ind[numPressure] = 2*nIdx + 1; 
                            numPressure++; //increment how many pressure bytes to expect
                        }  


                        
                        #ifdef DebugPrint
                            // for communication
                            strcpy(debugstr,"Inside Assignment");
                            TransmitPayload(HeaderStr,1,strlen(debugstr),debugstr,TrailerStr);
                        #endif

                        
                        PortCommand = PortCommand + "," + String(PortA[2*nIdx].action) + "," + String(PortA[2*nIdx+1].action) ;
                        


                        
                    }

                    else if (nIdx>=numExpectedBytes and nIdx< (numExpectedBytes+numPressure))
                    {

                        int pressportIdx = PressVal_ind[nIdx-numExpectedBytes]; //get the pressure value
                        PortA[pressportIdx].pressure = 12.0*Payload[nIdx]/255.0; //convert to a pressure value

                        #ifdef DebugPrint
                            // for communication
                            char pressure_s[10];
                            dtostrf(PortA[pressportIdx].pressure, 3, 4, pressure_s);
                            PressureCommand = PressureCommand + "," + pressure_s ;

                            strcpy(debugstr,"Inside Pressure Assignment");
                            TransmitPayload(HeaderStr,1,strlen(debugstr),debugstr,TrailerStr);
                        #endif

                    }

                    nIdx++;

                    if (nIdx == numExpectedBytes+numPressure) //if you get the number of bytes you expected, then move on to the next step
                    {

                        numPressure = 0;
                        nIdx = 0;
                        stateRx = ST_Trailer;

                        #ifdef DebugPrint
                            strcpy(debugstr,"Reset after Pressure Assignment");
                            TransmitPayload(HeaderStr,1,strlen(debugstr),debugstr,TrailerStr);
                        #endif
                    }

                    //update the pressure port actions.

                break;

                case ST_Trailer: //check to see if Trailer has been received properly
                    #ifdef DebugPrint
                        strcpy(debugstr,"Inside Trailer");
                        TransmitPayload(HeaderStr,1,strlen(debugstr),debugstr,TrailerStr);
                    #endif
                    

                    if (cByte == TrailerStr[nIdx])
                    {
                        nIdx++;

                        #ifdef DebugPrint
                            strcpy(debugstr,"Recognize Trailer char");
                            TransmitPayload(HeaderStr,1,strlen(debugstr),debugstr,TrailerStr);
                        #endif

                        // char val[3];
                        // val[0] = (char) (cByte);
                        // val[1] = "\0";

                        //TransmitPayload(HeaderStr,1,strlen(val),val,TrailerStr);

                        if( nIdx == MSG_TRLR_LEN)
                        {
                            stateRx = ST_Header;
                            nIdx = 0;

                            //Send the read pressure values
                            TransmitPressureReadings();

                            #ifdef DebugPrint

                                //Communicate the commanded pressures

                                strcpy(debugstr,"Ready to Transmit");
                                TransmitPayload(HeaderStr,1,strlen(debugstr),debugstr,TrailerStr);


                                char str[PressureCommand.length() + 1] = {};
                                strcpy(str, PressureCommand.c_str()); //convert to c_string
                                TransmitPayload(HeaderStr,1,strlen(str),str,TrailerStr);


                                char strPort[PortCommand.length() + 1] = {};
                                strcpy(strPort, PortCommand.c_str()); //convert to c_string
                                TransmitPayload(HeaderStr,1,strlen(strPort),strPort,TrailerStr);

                                PortCommand ="";


                                PressureCommand="";
                            

                                //Communicate that we are finished
                                strcpy(debugstr,"Finished");
                                TransmitPayload(HeaderStr,1,strlen(debugstr),debugstr,TrailerStr);

                            #endif
                        }
                    }

                    else
                    {
                        nIdx = 0;
                        stateRx = ST_Header;
                        //trailer not received propertly

                        #ifdef DebugPrint
                            strcpy(debugstr,"Trailer not received properly");
                            TransmitPayload(HeaderStr,1,strlen(debugstr),debugstr,TrailerStr);
                        #endif
                    }

                break;

            }

            //execute the pressure port actions in case you are stuck in the Serial.available()
            executePressurePortActions();

        }

    }

}


void executePressurePortActions()
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

    
    String outputs = "";
    for (int i = 0; i < numPressurePorts; i++)
    {

        PressurePort::PressureState PAction = PressurePort::HOLD;
        float activation = 0;

        switch (PortA[i].action)
        {
            case HOLD:
                Pa[i].setValve_state(PressurePort::HOLD,bitArray,MaxNumShiftRegisters, ArrayElementSize);  //keep valves closed


            break;

            case INFLATE:
                Pa[i].setValve_state(PressurePort::INFLATE,bitArray,MaxNumShiftRegisters, ArrayElementSize);  //keep valves closed

            break;

            case VACUUM:
                Pa[i].setValve_state(PressurePort::VACUUM,bitArray,MaxNumShiftRegisters, ArrayElementSize);  //keep valves closed

            break;

            case START:
                Pa[i].setValve_state(PressurePort::START,bitArray,MaxNumShiftRegisters, ArrayElementSize);  //keep valves closed

            break;

            case OPEN:
                Pa[i].setValve_state(PressurePort::OPEN,bitArray,MaxNumShiftRegisters, ArrayElementSize);  //keep valves closed
            break;

            case INFLATE_AND_STOP:
                activation = PortA[i].pressure/30;
                Pa[i].modulatePressure_Activation_STOP(activation, bitArray, MaxNumShiftRegisters, ArrayElementSize,true);  //writes to bitArray which valves to open or close
            break;
            
            case INFLATE_AND_MODULATE:
                activation = PortA[i].pressure/30;
                Pa[i].modulatePressure_Activation(activation, bitArray, MaxNumShiftRegisters, ArrayElementSize);  //writes to bitArray which valves to open or close
            break;

            case IGNORE:
            break;

            
            
        }
        
        float rawPress = Pa[i].readPressure(false);  //without moving average
        PortA[i].readPressure_psi = rawPress; //update the struct with the pressure value

        #ifdef DebugPrint
            char buffer[80];
            char pressure_s[10];
            dtostrf(rawPress, 3, 4, pressure_s);
            dtostrf(rawPress, 3, 4, pressure_s);
            outputs = outputs + "," + pressure_s ;
        #endif
        
       
    }

    

    
    PressurePort::WriteToShiftRegister(PressurePort::RCK, PressurePort::G_ENABLE, bitArray, MaxNumShiftRegisters); //executes the bitArray to the shift registers to actually open or close the valves 
    endTime = micros();
    
    #ifdef DebugPrint
        char str[outputs.length() + 1] = {};
        strcpy(str, outputs.c_str()); //convert to c_string
        TransmitPayload(HeaderStr,1,strlen(str),str,TrailerStr);
    #endif

}


void TransmitPressureReadings()
{
    byte bytePressure[numPressurePorts*4];

    for (int i=0;i<numPressurePorts;i++)
    {
        union byteUnion {  
            float f;  
            byte fByte[sizeof(float)];  
        };
        
        byteUnion bU;  
         
        bU.f = PortA[i].readPressure_psi;  
        for(int j=0;j<sizeof(float);j++)  
        {
            bytePressure[i*4+j] = bU.fByte[j]; 
        }
    }

    TransmitPayload(HeaderStr,0,numPressurePorts*4,bytePressure,TrailerStr); //protocol 0, i.e. expects floats
}

void TransmitPayload(char startChar[], byte protocolType,byte numBytes,  byte payload[], char endChar[] )
{

    unsigned char buffer[255];
    buffer[0]=startChar[0];
    buffer[1]=startChar[1];
    buffer[2]=protocolType;
    buffer[3] = numBytes;
    for (int i = 0;i<numBytes;i++)
    {
        buffer[4+i] = payload[i];
    }
    buffer[numBytes+4] = endChar[0];
    buffer[numBytes+5] = endChar[1];

    unsigned int len = 6+numBytes;


    Serial.write(buffer,len);

}