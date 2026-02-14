#include <Wire.h>

//--- Communication ---- -> Written with Help from ChatGPT

#define HEADER1 0xAA
#define HEADER2 0x55

const int IN_FLOATS = 2;
const int OUT_FLOATS = 3;

enum State { WAIT_HEADER1, WAIT_HEADER2, WAIT_LENGTH, WAIT_PAYLOAD, WAIT_CHECKSUM };
State rxState = WAIT_HEADER1;

byte rxLen = 0;
byte rxBuf[32];
byte rxCount = 0;
byte rxChecksum = 0;

byte seqCounter = 0;

bool frameReady = false;

//--- Position Control ----
int enable1Pin = 2;
int in1Pin = 0;
int in2Pin = 1;
int enable2Pin = 5;
int in3Pin = 3;
int in4Pin = 4;
int potPin1 = 38;
int potPin2 = 39;
int cur_pos1 = 0; // current position of the motor. 0 is all the way retracted, 1023 is all the way forward
int cur_pos2 = 0; // current position of motor 2. 0 is all the way retracted. 1023 is all the way forward
float kp = 10;
float ki = 0.001;
float kd = 0.1;
// initial values for PID calculations
float prevError = 0; // initial values for PID calculations
float sumError = 0; // integral term
float changeError = 0; // derivative term
float current_time = 0;
float prev_time = 0;
float time_transmission_ms = 20;
float pi = PI;
float des_pos1 = 0.0;
float des_pos2 = 0.0;
float previousMillis = 0;


//-------------- Load Cell ---------------//
union ByteToInt {
  byte arr_v[2];
  uint32_t int_v;
};

float force_reading = 0.0;




void setup() {

  Wire.begin();
  Serial.begin(115200);
  pinMode(in1Pin, OUTPUT);
  pinMode(in2Pin, OUTPUT);
  pinMode(enable1Pin, OUTPUT);
  pinMode(in3Pin, OUTPUT);
  pinMode(in4Pin, OUTPUT);
  pinMode(enable2Pin, OUTPUT);

 
}


void loop() {
  
  current_time = millis();
   // ---- Non-blocking receive ----
  while (Serial.available() > 0) {
    byte b = Serial.read();

    switch (rxState) {
      case WAIT_HEADER1:
        if (b == HEADER1) rxState = WAIT_HEADER2;
        break;
      case WAIT_HEADER2:
        if (b == HEADER2) rxState = WAIT_LENGTH;
        else rxState = WAIT_HEADER1;
        break;
      case WAIT_LENGTH:
        rxLen = b;
        if (rxLen <= sizeof(rxBuf)) {
          rxCount = 0;
          rxChecksum = 0;
          rxState = WAIT_PAYLOAD;
        } else {
          rxState = WAIT_HEADER1;
        }
        break;
      case WAIT_PAYLOAD:
        rxBuf[rxCount++] = b;
        rxChecksum ^= b;
        if (rxCount >= rxLen)
          rxState = WAIT_CHECKSUM;
        break;
      case WAIT_CHECKSUM:
        if (b == rxChecksum) {
          frameReady = true;
        }
        rxState = WAIT_HEADER1;
        break;
    }
  }

  // ---- Process complete frame ----
  if (frameReady) {
    frameReady = false;
    if ((current_time - prev_time)>(time_transmission_ms))
      {
        prev_time = millis();
        processCommand(rxBuf, rxLen);
      }
  }
  // Other tasks can run freely here (non-blocking)
  readForce();
  manageMotors();

}

void readForce()
{
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

  // Reverse Bit Order
  for (int j = 0;j<2;j++)
  {
    sensorB.arr_v[j] = num[1-j];
  }

  //Serial.println(sensorB.int_v, HEX);
  //Serial.println(sensorB.int_v, DEC);

  force_reading = sensorB.int_v/1.0;  //save value as a float.
}

void manageMotors()
{
  cur_pos1 = analogRead(potPin1);
  delay(2);
  cur_pos1 = analogRead(potPin1);
  delay(2);

  //Serial.println("Successfully read position 1");

  cur_pos2 = analogRead(potPin2);
  delay(2);
  cur_pos2 = analogRead(potPin2);
  //Serial.println("Successfully read position 2");

  
  setMotor(in1Pin, in2Pin, enable1Pin, des_pos1, cur_pos1, kp, ki, kd);
  //Serial.println("Successfully activated actuator 1");
  setMotor(in3Pin, in4Pin, enable2Pin, des_pos2, cur_pos2, kp, ki, kd);
}

void setMotor(int in1Pin, int in2Pin, int enablePin, float des_pos, float cur_pos, float kp, float ki, float kd)
{
  float error = des_pos-cur_pos;
  if (abs(error)<10) //threshold to ignore error
  {
    error = 0;
  }
  // Serial.print("Error:");
  // Serial.println(error);


  double currentMillis = millis();
  double elapsedTime = (currentMillis - previousMillis) / 1000.0; // Convert to seconds
  previousMillis = currentMillis;

  sumError += error*elapsedTime; // update integral term
  changeError = error - prevError; // update derivative term
  int speed = kp*(error) + ki*(sumError) + kd*(changeError); // PID calculation 
  // Serial.println(speed);
  
  prevError = error; // update prevError
  boolean reverse = false;

  if (error<0) //if cur_pos is greater than des_pos, then you need to reverse
  {
    reverse = true;
  }

  speed = max(min(abs(speed),255),0);
  // Serial.print("Speed:");
  // Serial.println(speed);
  // Serial.println("");

  // Serial.println(reverse);
  analogWrite(enablePin, speed);
  digitalWrite(in1Pin, reverse);
  digitalWrite(in2Pin, !reverse);
}

void processCommand(byte *payload, byte len) {
  if (len != IN_FLOATS * 4) return; // sanity check

  float targetPos[IN_FLOATS];
  memcpy(targetPos, payload, len);
  des_pos1 = targetPos[0];
  des_pos2 = targetPos[1];
  // Simulate reading sensors
  float sensor1 = cur_pos1;
  float sensor2 = cur_pos2;




  float force   = force_reading;

  float outData[OUT_FLOATS] = {sensor1, sensor2, force};

  sendFrame((byte*)outData, sizeof(outData));
}



// void sendFrame(byte *payload, byte len) {
//   byte checksum = 0;
//   byte frame[40];
//   int index = 0;
//   frame[index++] = HEADER1;
//   frame[index++] = HEADER2;
//   frame[index++] = len + 1; // +1 for seq byte
//   frame[index++] = seqCounter; // sequence number

//   for (int i = 0; i < len; i++) {
//     frame[index++] = payload[i];
//     checksum ^= payload[i];
//   }
//   checksum ^= seqCounter;

//   frame[index++] = checksum;
//   Serial.write(frame, index);

//   seqCounter++; // increment each frame
// }

void sendFrame(byte *payload, byte len) {
  byte checksum = 0;
  for (int i = 0; i < len; i++) checksum ^= payload[i];

  Serial.write(HEADER1);
  Serial.write(HEADER2);
  Serial.write(len);
  Serial.write(payload, len);
  Serial.write(checksum);
}

