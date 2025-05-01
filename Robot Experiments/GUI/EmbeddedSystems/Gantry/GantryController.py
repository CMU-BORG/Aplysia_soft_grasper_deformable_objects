
import pathlib
import re
import time
from datetime import datetime
from enum import Enum

import numpy as np


#import torch
#from .envs.GantrySimulation import GantrySimulation
#from ..SNS.controller.SNS_layer import perceptor, controller


import serial
from ..Support.Structures import Point, Velocity, Acceleration


import logging
from datetime import datetime
import sys

from pathlib import Path


class GantryActions(Enum):
    STAY = 0 #dont move
    MOVE = 1 #Move jaws closer together

class Gantry:

    def __init__(self,comport='COM12',serialRate=115200,timeout=2,initPos=Point(226,252,200),MaxBufferSize = 3,MoveSpeed_mm_p_min = 50*60, homeSystem = True, initializeSystem = True):

        self.loggerGC = None
        self.setupLogger()


        self.ser = None
        self.initPos = initPos #position in [x,y,z] in mm. x is movement of the gantry head, y is the movement of the base plate and z is the movement up and down.

        self.maxPos_mm = Point(450, 480, 300) #max position from the endstops in mm for x,y and z, respectively
        self.BufferLength = 0
        self.MaxBufferSize = MaxBufferSize #maximum number of commands to keep in the buffer
        self.MoveSpeed = MoveSpeed_mm_p_min #mm/min
        self.MaxSpeedRate_mmps = Velocity(300, 300 , 35) #mm/s.  Max feedrate for the x, y and z axes.
        self.MaxAcceleration_mmps2 = Acceleration(500, 500, 100) #mm/s^2.  Max acceleration for the x, y and z axes
        self.PositionArray = {"time":[0],"x":[0],"y":[0],"z":[0]} #indicates current position
        self.goalPos = Point(0, 0, 0) #triad indicating goal position
        self.inMotion = False #set to true if the gantry is executing a linear move, false otherwise
        self.atGoal = False #set to true if the gantry head is at the goal, false otherwise
        self.readyToInject = True #set to true if it is ready to take another command, false otherwise. it is ready to take a command when it is at goal and not in motion, or in motion and >75% of the distance from the starting pos to the goal position there.
        self.GoalTolerance_mm = 0.5 #distance away from the goal in mm to be considered "at goal"
        self.startMotionPos = Point(0, 0, 0) #triad indicating where the current motion was started from

        self.feedrate_mmps = self.MoveSpeed/60

        #Run Initialization routine:
        if initializeSystem == True:
            self.SetupGantry(comport = comport, serialRate = serialRate,timeout=timeout,initPos = initPos, homeSystem=homeSystem)

    def setupLogger(self):
        ##### Set up logging ####
        loggerGC = logging.getLogger(__name__)
        fname = Path(__file__).parents[3].joinpath('datalogs', str(__name__) + datetime.now().strftime(
            "_%d_%m_%Y_%H_%M_%S") + ".txt")

        fh = logging.FileHandler(fname)  # file handler
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(sys.stdout)  # stream handler
        ch.setLevel(logging.ERROR)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        loggerGC.setLevel(logging.DEBUG)

        # add the handlers to the loggerGC
        loggerGC.addHandler(fh)
        loggerGC.addHandler(ch)

        self.loggerGC = loggerGC
    def SetupGantry(self,comport='COM12',serialRate=115200,timeout=2,initPos=Point(220,220,315), homeSystem = True):
        self.ser = serial.Serial(comport, serialRate, timeout=timeout)
        time.sleep(20)

        if homeSystem == True:
            self.HomeGantry(initPos)

    def HomeGantry(self,initPos:Point):
        self.loggerGC.info("Homing gantry ...")
        self.sendCommand("G28 X0 Y0 Z0\r\n")
        # sendCommand(ser,"G0 F15000 X0\r\n")
        self.sendCommand("M400\r\n")
        time.sleep(2)

        self.sendCommand("G90\r\n")
        self.loggerGC.debug("Finished Sending G90")
        time.sleep(1)
        self.sendCommand("G0 F15000\r\n")
        time.sleep(1)
        self.sendCommand("G0 F5000 " + 'X{0} Y{1} Z{2}'.format(*initPos) + "\r\n")  # move to offset position
        self.sendCommand("M400\r\n")  # waits until all motions in the planning queue are completed
        time.sleep(1)
        self.sendCommand("M280 P0 S95\r\n")  # make sure that the gripper is closed
        self.sendCommand("M400\r\n")
        self.loggerGC.info("Finished homing gantry ...")


    def sendCommand(self,commandstr,wait_for_ok = True):
        self.ser.write(commandstr.encode('utf-8'))
        # readval = ser.read(20)

        # print(readval)
        line_hold=""
        while wait_for_ok:
            line = self.ser.readline()
            self.loggerGC.debug('Wait for ok: '+str(line,'utf-8'))
            line_hold = line_hold + str(line, 'utf-8') +"\n"

            if line == b'ok\n':
                break

        return(line_hold)

    def getPosition(self):
        try:
            tic = time.perf_counter()
            responseM114 = self.sendCommand("M114 B\r\n")
            M114_restr = "X:(?P<Xcom>\d*\.\d*).*Y:(?P<Ycom>\d*\.\d*).*Z:(?P<Zcom>\d*\.\d*).*E:(?P<Ecom>\d*\.\d*).*B:(?P<BufferCountHead>\d*)\s*(?P<BufferCountTail>\d*).*Count.*X:(?P<Xsteps>\d*).*Y:(?P<Ysteps>\d*).*Z:(?P<Zsteps>\d*)"
            matchM114 = re.search(M114_restr, responseM114)
            matchM114_dict = matchM114.groupdict()
            Xcom, Ycom, Zcom, Ecom = float(matchM114_dict['Xcom']), float(matchM114_dict['Ycom']), float(
                matchM114_dict['Zcom']), float(
                matchM114_dict['Ecom'])  # not real time positons, but what was commanded last
            Xsteps, Ysteps, Zsteps = int(matchM114_dict['Xsteps']), int(matchM114_dict['Ysteps']), int(
                matchM114_dict['Zsteps'])  # position in steps
            BufferCountHead = int(matchM114_dict['BufferCountHead'])  # buffer size is 16 (0 indexed, 0:15)
            BufferCountTail = int(matchM114_dict['BufferCountTail'])
            self.BufferLength = (
                BufferCountHead - BufferCountTail if BufferCountHead >= BufferCountTail else BufferCountHead - BufferCountTail + 16) #update the buffer length

            x_off = self.initPos[0]
            y_off = self.initPos[1]
            z_off = self.initPos[2]

            x, y, z = (Xsteps / 80 - x_off) / 1000, (Ysteps / 80 - y_off) / 1000, (
                    Zsteps / 400 - z_off) / 1000  # convert from steps to m

            #update the time and the positions
            positionArray =[x,y,z]
            self.PositionArray["time"].append(datetime.now())
            self.PositionArray["x"].append(x*1000) #multiply by 1000 to convert from m to mm
            self.PositionArray["y"].append(y*1000)
            self.PositionArray["z"].append(z*1000)

            #Check if still in motion
            commandedPos = Point(Xcom*1000,Ycom*1000,Zcom*1000)
            inMotionBool = self.InMotionCheck(currentPosition=Point(*positionArray),commandedPosition = commandedPos)

            #Check if at goal
            atGoalBool = self.atGoalCheck(currentPosition= Point(*positionArray), goalPosition = self.goalPos)

            #Update Inject status based on size of the buffer
            injectBool = self.updateInjectStatus()

            toc = time.perf_counter()
            self.loggerGC.debug("M114 Time " + str(toc - tic))

            return(Point(*positionArray))

        except Exception as e:
            self.loggerGC.exception(responseM114)
            return([np.Inf,np.Inf,np.Inf]) #return Inf for all three axes

    def CheckPositionToReference(self, position1: Point, position2: Point, tolerance: float):  # true if points are within tolerance of each other , false if outside tolerance
        distanceval = np.linalg.norm([np.abs(x - position2[i])  for (i, x) in
                      enumerate(position1)])
        truthValue = distanceval <= tolerance
        return (truthValue,distanceval)

    def InMotionCheck(self,currentPosition: Point, commandedPosition: Point): #true if still in motion, false if not moving
        MotionBool, distanceval = (self.CheckPositionToReference(currentPosition,commandedPosition,self.GoalTolerance_mm)) #negate result because if still in motion, it is outside the commanded
        MotionBool = not(MotionBool)
        self.inMotion = MotionBool
        return(MotionBool)

        # truthArray = [np.abs(x-commandedPosition[i])<=self.GoalTolerance_mm for (i,x) in enumerate(currentPosition)]
        # if np.all(truthArray) == True:
        #     self.inMotion = False
        #     return (False)
        #
        # else:
        #     self.inMotion = True
        #     return (True)


    def atGoalCheck(self, currentPosition: Point, goalPosition: Point): #true if at goal within tolerance, false otherwise.  2nd argument returned is fraction of distance covered
        GoalBool, distanceval = (self.CheckPositionToReference(currentPosition, goalPosition, self.GoalTolerance_mm))

        distanceGoalToStart = np.linalg.norm([x-self.startMotionPos[i] for (i,x) in enumerate(goalPosition)]) #get distance from goal to start

        if distanceGoalToStart ==0 and distanceval == 0:
            fractionCovered = 1

        elif distanceGoalToStart == 0:
            fractionCovered = 1 #need to fix this

        else:
            fractionCovered = distanceval/distanceGoalToStart


        self.atGoal = GoalBool
        return (GoalBool,fractionCovered)

    def updateInjectStatus(self):#fraction distance covered is fraction of the starting distance covered.  #threshold is the percentage covered
        if self.BufferLength < self.MaxBufferSize:
            self.readyToInject = True
        else:
            self.readyToInject = False

        return(self.readyToInject)


    def setXYZ_Position(self,x_mm,y_mm,z_mm,MoveSpeed=None):

        x_off = self.initPos[0]
        y_off = self.initPos[1]
        z_off = self.initPos[2]


        if MoveSpeed is None:
            MoveSpeed = self.MoveSpeed

        posvec_positions = [x_mm + x_off, y_mm + y_off,
                  z_mm + z_off] 

        posvec_positions = [max(min(x, self.maxPos_mm[i]), 0) for (i, x) in enumerate(posvec_positions)] #limit between 0 and maximum allowable move for that axis
        posvec = [MoveSpeed,*posvec_positions]  # first element is the speed in mm/min

        commandStr = "G0 F{0:.2f} X{1:.2f} Y{2:.2f} Z{3:.2f} \r\n".format(*posvec)
        if self.BufferLength < self.MaxBufferSize:  # only send commands when the buffer is small
            # print("Buffer "+str(BufferLength))
            # print(commandStr)

            self.sendCommand(commandStr)




    def calculateIncrementalMove(self,move_x_mm = 0, move_y_mm = 0, move_z_mm = 0, TrackPosition = False): #x, y and z are increments to move, not absolute position

        if TrackPosition == True:
            curPos = self.getPosition() #get current position
        else:
            curPos = self.goalPos
        posVec = [move_x_mm, move_y_mm, move_z_mm]

        if self.readyToInject == True & np.any([x !=0 for x in posVec]): #only update startMotionPos if you are are ready to inject and one or more of the commanded axes increments is non zero
            self.startMotionPos = curPos # note the starting position of the move so that later we can check how far into the move we are.
            self.goalPos = [max(min(x+self.initPos[i]+self.goalPos[i], self.maxPos_mm[i]), 0)-self.initPos[i] for (i, x) in enumerate(posVec)] #Update the goal position. limit the move to between 0 and the maximum, but first need to convert incremental move to absolute move, and then afterwards subtract the offset




    def incrementalMove(self,move_x_mm = 0, move_y_mm = 0, move_z_mm = 0, moveSpeed_mmps =None): #x, y and z are increments to move, not absolute position

        curPos = self.getPosition() #get current position
        posVec = [move_x_mm, move_y_mm, move_z_mm]

        if self.readyToInject == True & np.any([x !=0 for x in posVec]): #only update startMotionPos if you are are ready to inject and one or more of the commanded axes increments is non zero
            self.startMotionPos = curPos # note the starting position of the move so that later we can check how far into the move we are.
            self.goalPos = [max(min(x+self.initPos[i]+self.goalPos[i], self.maxPos_mm[i]), 0)-self.initPos[i] for (i, x) in enumerate(posVec)] #Update the goal position. limit the move to between 0 and the maximum, but first need to convert incremental move to absolute move, and then afterwards subtract the offset


        #newPos = [max(min(x+self.initPos[i]+curPos[i], self.maxPos_mm[i]), 0)-self.initPos[i] for (i, x) in enumerate(posVec)] #limit the move to between 0 and the maximum, but first need to convert incremental move to absolute move, and then afterwards subtract the offset


        self.setXYZ_Position(*self.goalPos, moveSpeed_mmps*60) #send commands to move to the new position


