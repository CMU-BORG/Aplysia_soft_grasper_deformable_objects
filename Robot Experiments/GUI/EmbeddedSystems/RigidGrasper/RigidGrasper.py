from enum import Enum

import numpy as np
import re
import struct
import sys


import serial
import math

import asyncio

import time

import logging
from datetime import datetime

from pathlib import Path

from dynamixel_sdk import *

class GrasperActions(Enum):
    STAY = 0 #dont move
    CLOSE = 1 #Move jaws closer together
    OPEN = 2 #Move jaws apart

class ForceSensor(Enum):
    IGNORE = 0 #don't do anything
    READ_FORCE = 1 #read force value

class RigidGrasper:

    def __init__(self,data_rate = 115200, port_num = "COM3", GoalPosition1=[0,1023], GoalPosition2 = [0,1023], useForceSensor = False):

        # setup Logger
        self.logger = None
        self.setupLogger()

        # Goal Positions for the Grasper:
        self.NumMotors = 2 #one Dynamixel each for left and right
        self.GoalPosition_Limits = {"1":GoalPosition1, "2":GoalPosition2} #limits of motion

        # Current position of the grasper:
        self.CurrentPosition ={"1":-1000000, "2":-1000000}
        self.CurrentDistance = 0

        # max displacement of jaws
        self.max_displacement = 63  # np.inf, in mm
        self.actuator_max_displacement = 25  # in mm, how much each individual actuator can move maximum
        self.d0 = 63  # in mm, how far the jaws are apart when the actuators are fully retracted

        # Commanded Position for the grasper
        self.commandedPosition = {"ClosureDistance_mm":self.max_displacement}

        # Communication
        self.HEADER1 = 0xAA
        self.HEADER2 = 0x55
        self.rx_state = "WAIT_HEADER1"
        self.rx_len = 0
        self.rx_buf = bytearray()
        self.rx_checksum = 0
        self.last_tx_time = 0

        # SetupMotors
        self.data_rate = data_rate
        self.port_num = port_num
        self.ser = None
        self.setupComms_and_Motors()


        #Force Sensor
        self.useForceSensor = useForceSensor
        if self.useForceSensor == True:
            self.numPorts = 1
            self.ForceArray = [[0] for x in range(0, self.numPorts)]
            self.RawForceArray = [[0] for x in range(0, self.numPorts)]
            self.PrevJawForce = [0 for x in range(0, self.numPorts)] #list to hold the original Jaw force
            self.changeInForce = 0 #change in force relative to baseline jaw force.





    def setupLogger(self):
        ##### Set up logging ####
        logger_soft = logging.getLogger(__name__)

        fname = Path(__file__).parents[3].joinpath('datalogs', str(__name__) + datetime.now().strftime(
            "_%d_%m_%Y_%H_%M_%S") + ".txt")

        fh = logging.FileHandler(fname)  # file handler
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler(sys.stdout)  # stream handler
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger_soft.setLevel(logging.DEBUG)
        # add the handlers to the logger_soft
        logger_soft.addHandler(fh)
        logger_soft.addHandler(ch)
        self.logger = logger_soft
    def setupComms_and_Motors(self):
        self.ser = serial.Serial(self.port_num, self.data_rate, timeout=0)
    def GetCountFromGripperWidth(self,gripperWidth_mm):


        gWL = (self.d0-gripperWidth_mm)/2
        gWR = (self.d0-gripperWidth_mm)/2



        M1_count = gWL*self.GoalPosition_Limits["1"][1]/self.actuator_max_displacement  #should be 1023*gwL/25
        M2_count = gWR*self.GoalPosition_Limits["2"][1]/self.actuator_max_displacement


        M1_count = np.clip(M1_count,self.GoalPosition_Limits["1"][0],self.GoalPosition_Limits["1"][1])
        M2_count = np.clip(M2_count, self.GoalPosition_Limits["2"][0], self.GoalPosition_Limits["2"][1])
        return(M1_count,M2_count)

    def IncrementalMove(self, moveIncrement_mm = 1, action1=GrasperActions.STAY,
                        action2=GrasperActions.STAY):  # close claws, assume position control

        #TODO: Limit the move so it doesn't get -ve distance -> kinda done but need better limits
        #CurrentPosition, dxl_comm_result, dxl_error = self.ReadCurrentPosition()  # get current position and update member variable with the same.

        # self.commandedPosition["ClosureDistance_mm"] = np.clip(self.commandedPosition["ClosureDistance_mm"] + moveIncrement_mm, 0, 100)
        self.commandedPosition["ClosureDistance_mm"] = np.clip(
            self.CurrentDistance - moveIncrement_mm, 0, 100) #the rigid grasper closes inwards and current distance is the the distance between the jaws, so for A to close the grasper, need to subtract the increment.

        # Read pressure sensor:
        if self.useForceSensor == True:
            self.MoveGrasper()
            self.ReadGrasperData()

    async def ReadSensorValues(self,number_avg = 1, loop_delay = 0.001, rawVal = False): #convenience function to get the pressure values
        jaw_sensor = np.array([0])
        jawForce = np.array([0])
        closureDistance = 0
        await asyncio.sleep(0.001)


        for i in range(number_avg):
            self.ReadGrasperData()

            #raw sensor values
            jaw_sensor_r = self.RawForceArray[0]
            jaw_sensor = jaw_sensor_r + jaw_sensor
            self.logger.debug("Read sensor values loop %i, jaw force raw %f" % (i, jaw_sensor_r))

            #converted force values
            jawForce_r = self.changeInForce
            self.logger.debug("Read sensor values loop %i, jaw force in N %f"%(i,*jawForce_r))
            jawForce = jawForce_r+jawForce

            #distance values
            closureDistance = self.CurrentDistance + closureDistance
            self.logger.debug("Closure distance loop %i, closure distance in mm %f" % (i, closureDistance))
            await asyncio.sleep(loop_delay)


        jaw_sensor = jaw_sensor/number_avg
        jawForce = jawForce/number_avg
        closureDistance = closureDistance/number_avg

        if rawVal == False: #return the force value in newtons
            return(jawForce,closureDistance)

        elif rawVal == True:
            return(jaw_sensor,closureDistance) #return the raw averaged readings and the closure distance

    def MoveGrasper(self): #close claws, assume position control

        M1_count,M2_count = self.GetCountFromGripperWidth(self.commandedPosition["ClosureDistance_mm"])
        target_positions = (M1_count, M2_count)
        if self.ser.in_waiting>20:
            self.ser.reset_input_buffer()
            time.sleep(0.001)
        #print(self.ser.in_waiting)
        self.send_floats(target_positions)



    def ReadGrasperData(self):

        #Read pressure sensor:
        self.MoveGrasper() #only sends data when a command is sent ...
        if self.useForceSensor == True:
            # Read Non-blocking
            CurrentPosition = {"1":0,"2":0}
            result = self.read_nonblocking()
            if result:
                s1, s2, force = result

                self.ser.flush()

                self.logger.debug(f"Sensor1={s1:.3f}, Sensor2={s2:.3f}, Force={force:.3f}")
                self.RawForceArray[0] = force
                ForceN = self.calcForceFromSensor(force)

                self.ForceArray[0] = [ForceN]

                CurrentPosition["1"] = s1
                CurrentPosition["2"] = s2

                self.logger.debug("Data for Case 0 (Force values in N): " + ','.join(str(ForceN)))

                self.CurrentPosition = CurrentPosition


            ChF = self.getJawChangeForce()
            self.changeInForce = ChF
            self.logger.debug("Change in Force: " + ','.join([str(x) for x in ChF]))
            self.CurrentDistance = self.ConvertPositionToDistance_mm(CurrentPosition["1"],CurrentPosition["2"])


    def ConvertPositionToDistance_mm(self, pos1, pos2):

        d1 = pos1*self.actuator_max_displacement/1023
        d2 = pos2*self.actuator_max_displacement/1023
        gW = np.clip(self.d0-(d1+d2), 0 ,np.Inf)
        return gW #TODO: update this

    # ---- Framing ----
    def _calc_checksum(self, data: bytes) -> int:
        c = 0
        for b in data:
            c ^= b
        return c

    def send_floats(self, floats):
        """Send a list/tuple of floats as a framed binary message"""
        payload = struct.pack('<' + 'f' * len(floats), *floats)
        checksum = self._calc_checksum(payload)
        frame = bytes([self.HEADER1, self.HEADER2, len(payload)]) + payload + bytes([checksum])
        self.ser.write(frame)


    def read_nonblocking(self):
        """Call periodically to process any incoming bytes.
           Returns a list of floats if a complete frame is received."""
        while self.ser.in_waiting > 0:
            b = self.ser.read(1)
            if not b:
                return None
            byte = b[0]

            if self.rx_state == "WAIT_HEADER1":
                if byte == self.HEADER1:
                    self.rx_state = "WAIT_HEADER2"
            elif self.rx_state == "WAIT_HEADER2":
                if byte == self.HEADER2:
                    self.rx_state = "WAIT_LENGTH"
                else:
                    self.rx_state = "WAIT_HEADER1"
            elif self.rx_state == "WAIT_LENGTH":
                self.rx_len = byte
                self.rx_buf = bytearray()
                self.rx_checksum = 0
                self.rx_state = "WAIT_PAYLOAD"
            elif self.rx_state == "WAIT_PAYLOAD":
                self.rx_buf.append(byte)
                self.rx_checksum ^= byte
                if len(self.rx_buf) >= self.rx_len:
                    self.rx_state = "WAIT_CHECKSUM"
            elif self.rx_state == "WAIT_CHECKSUM":
                if byte == self.rx_checksum:
                    # Valid frame — unpack floats
                    if len(self.rx_buf) % 4 == 0:
                        count = len(self.rx_buf) // 4
                        floats = struct.unpack('<' + 'f' * count, self.rx_buf)
                        self.rx_state = "WAIT_HEADER1"
                        return floats
                # Invalid or done: reset
                self.rx_state = "WAIT_HEADER1"
        return None  # no complete frame yet



    def calcForceFromSensor(self, reading):
        Force_N = (0.00361*reading-15)
        Force_N = np.clip(Force_N,-20,40)
        return(Force_N)


    def getJawChangeForce(self):
        if (len(self.ForceArray[0]) <= 0):
            return ([0])
        else:

            CurJawForce = [self.ForceArray[0][-1]]
            # PrevJawPress= [self.PressureArray[x][-2] for x in self.JawPos]
            # if self.PrevJawForce is None:
            #     self.PrevJawForce = CurJawForce
            #     self.logger.info('baseline Jaw force is ' + str(self.PrevJawForce))

            ChangeInForce = (np.array(CurJawForce) - np.array(self.PrevJawForce)).tolist()
            # return(ChangeInPressure)
            return (ChangeInForce)


def CyclicTestGrasper(self):

        index = 0
        while 1:
            print("Press any key to continue! (or press ESC to quit!)")
            if getch() == chr(0x1b):
                break

            # Write goal position for Motor 1
            self.SetGoalPosition(self.GoalPosition_Limits["1"][index],self.GoalPosition_Limits["2"][index])


            MotorFin = {"1":False,"2":False}

            while 1:

                # Read present position
                CurrentPosition, dxl_comm_result, dxl_error = self.ReadCurrentPosition()
                print(CurrentPosition)

                for i,(k,currPos) in enumerate(CurrentPosition.items()):

                    if abs(self.GoalPosition_Limits[k][index] - currPos) > self.DXL_MOVING_STATUS_THRESHOLD:
                        if dxl_comm_result != COMM_SUCCESS:
                            print("%s\n" % packetHandler.getTxRxResult(dxl_comm_result))
                        elif dxl_error != 0:
                            print("%s\n" % packetHandler.getRxPacketError(dxl_error))

                        print("[ID:%03d] GoalPos:%03d  PresPos:%03d\n" % (self.DXL_ID[k],
                                                                          self.GoalPosition_Limits[k][index], currPos))

                    else:
                        MotorFin[k] =True

                if np.all(np.array(list(MotorFin.values()))==True):
                    break

            # Change goal position
            if index == 0:
                index = 1
            else:
                index = 0

if __name__ == '__main__':
    RG = RigidGrasper(data_rate = 115200, port_num = "COM7", useForceSensor = True)
    time.sleep(2)

    while  (True):
        RG.commandedPosition["ClosureDistance_mm"] = 42
        RG.MoveGrasper()

        RG.ReadGrasperData()

        CurrentPosition = RG.CurrentPosition
        print("Commanded pos: %i,  Current Pos: %i,%i." % (
            RG.commandedPosition["ClosureDistance_mm"],
            CurrentPosition["1"], CurrentPosition["2"]))


        print(RG.changeInForce)

        time.sleep(0.001)

    # while(input("Press a key to increment by 100")):
    #     RG.IncrementalMove(moveIncrement1 = 50,moveIncrement2 = 50, action1 = GrasperActions.CLOSE,action2 = GrasperActions.CLOSE)
    #     time.sleep(3)
    #     #RG.ReadCurrent()
    #     CurrentPosition,dxl_comm_result,dxl_error = RG.ReadCurrentPosition()
    #     print("%i,%i. In Deg: %f, %f:"%(CurrentPosition["1"],CurrentPosition["2"],CurrentPosition["1"]*360/4096,CurrentPosition["2"]*360/4096))



# RG = RigidGrasper()
# RG.CyclicTestGrasper()




