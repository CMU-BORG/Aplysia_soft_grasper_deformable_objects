#include <Wire.h>
#include <Bounce.h>










//--------------Loop Timing--------------- //

unsigned long startTime = 0;     //uS
unsigned long endTime = 0;       //uS
unsigned long previousTime = 0;  //ms
unsigned long LoopTime = 1000;   //us
unsigned long t_init = 0;        //time when the loop starts to begin calculating sinusoid

IntervalTimer PressureControlTimer;

int count = 0;  //how many times through pressure control loop
const int numForcePorts = 1;  


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
const int numExpectedBytes = 1;

enum PortAction_ENUM
{
    IGNORE = 0,
    READ_FORCE = 1,
};

byte Payload[MSG_PAYLOAD_LEN_MAX];

String PressureCommand;
String PortCommand;

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

//-------Function defs---------:
void runActivationPattern();
void read_Commands();
void ReadSerial();
void executeForcePortActions();
void TransmitPayload(char startChar[], byte protocolType,byte numBytes,  byte payload[], char endChar[] );
void TransmitForceReadings();

void setup() {
  // put your setup code here, to run once:
    Serial.begin(460800);  //apparently can communicate at 1 MBaud 
    Wire.begin();

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
    executeForcePortActions();

    
    //delayMicroseconds(1000);

}

void ReadSerial()
{ //inspired by: https://forum.arduino.cc/t/proper-method-of-receiving-uart-packet-data-far-exceeding-rx-buffer-size/611982/8


    byte cByte;

    static int nIdx = 0;
    static int numForce = 0;

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
                        if (Payload[nIdx] == READ_FORCE)
                        {
                          PortA.action = READ_FORCE;
                            
                        }


                        
                        #ifdef DebugPrint
                            // for communication
                            strcpy(debugstr,"Inside Assignment");
                            TransmitPayload(HeaderStr,1,strlen(debugstr),debugstr,TrailerStr);
                        #endif

                        
                        PortCommand = PortCommand + "," + String(PortA.action) ;
                        


                        
                    }

                    
                    nIdx++;

                    if (nIdx == numExpectedBytes+numForce) //if you get the number of bytes you expected, then move on to the next step
                    {

                        numForce = 0;
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

                            if (PortA.action == READ_FORCE)
                            {
                              //Send the read pressure values
                              TransmitForceReadings();
                            }

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
            executeForcePortActions();

        }

    }

}


void executeForcePortActions()
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

    switch (PortA.action)
    {
        case READ_FORCE:
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


        break;

        case IGNORE:
        break;
        
    }

    #ifdef DebugPrint
        char buffer[80];
        char pressure_s[10];
        dtostrf(PortA.readADC, 3, 4, pressure_s);
        outputs = pressure_s ;
    #endif
        


    endTime = micros();
    
    #ifdef DebugPrint
        char str[outputs.length() + 1] = {};
        strcpy(str, outputs.c_str()); //convert to c_string
        TransmitPayload(HeaderStr,1,strlen(str),str,TrailerStr);
    #endif

}


void TransmitForceReadings()
{
    byte byteForce[numForcePorts*4];

    for (int i=0;i<numForcePorts;i++)
    {
        union byteUnion {  
            float f;  
            byte fByte[sizeof(float)];  
        };
        
        byteUnion bU;  
         
        bU.f = PortA.readADC;  
        for(int j=0;j<sizeof(float);j++)  
        {
            byteForce[i*4+j] = bU.fByte[j]; 
        }
    }

    TransmitPayload(HeaderStr,0,numForcePorts*4,byteForce,TrailerStr); //protocol 0, i.e. expects floats
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