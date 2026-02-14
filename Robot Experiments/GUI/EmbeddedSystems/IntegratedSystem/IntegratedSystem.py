import time
import asyncio
import aioconsole
import numpy as np

import logging
from datetime import datetime
from tkinter import filedialog
import sys

from pathlib import Path

from enum import Enum

from GUI.EmbeddedSystems.SoftGrasper.SoftGrasper import PortActions
from GUI.EmbeddedSystems.SoftGrasper.SoftGrasper import SoftGrasper
from GUI.EmbeddedSystems.RigidGrasper.RigidGrasper import RigidGrasper
from GUI.EmbeddedSystems.Gantry.GantryController import Gantry as GantryController
import GUI.EmbeddedSystems.JoyCon.JoyCon as JC
from GUI.EmbeddedSystems.SNS.SNScontroller import SNScontroller, ControlType
from GUI.EmbeddedSystems.Support.Structures import GrasperContactForce,Point
from copy import deepcopy

import aioconsole

import cv2 as cv

from GUI.EmbeddedSystems.Koopman.KoopmanTesting import  koopman as kpm, variable_type
from GUI.EmbeddedSystems.Controllers.LQR import LQR


class GrasperType(Enum):
    SoftGrasper = 0 #Soft Grasper
    RigidGrasper = 1 #Rigid Grasper


### For items for user study experiments ####
class itemStatus(str,Enum):
    notstarted = 'notstarted'
    inprogress = 'inprogress'
    fail = 'fail'
    success = 'success'
    broken = 'broken'

class itemClass(Enum):
    fixed= 'fixed'
    breakable = 'breakable'
    graspable = 'graspable'
    notspecified = 'notspecified'

class itemTrack:
    def __init__(self,objectID, status = itemStatus.notstarted,itemClass=itemClass.graspable):
        self.objectID = objectID
        self.status = status #Not started, inprogress, fail, broken, success
        self.itemClass = itemClass #fixed, breakable, graspable

### End items for user study experiments ####

class IntegratedSystem:
    def __init__(self):
        self.SG = None # Soft Grasper or rigid grasper
        self.GC = None # Gantry Controller
        self.SNSc = None # SNS
        self.jcSG = None #Joycon+Soft Grasper controller

        self.grasperType = None

        #calibration parameters
        self.calibrationParams = {"Calibration Distance (mm)":1, "Grasp Pressure Threshold (psi)":[0.005,0.005,0.005], "Grasp Lift Height (mm)":35,
                                  "Touch Object Distance (mm)": 0.1, "Touch Pressure Threshold (psi)":[0.005,0.005,0.005] }


        #object for user experiments
        # object 1 is top left, object 9 is bottom right.
        self.ObjectVal = {"1": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
                     "2": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
                     "3": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
                     "4": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
                     "5": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
                     "6": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
                     "7": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
                     "8": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
                     "9": itemTrack(1, itemStatus.notstarted, itemClass.notspecified)
                     }

        self.fragileThreshold = 0.4 #scale of 0 to 1.  Forces above this are considered broken for fragile objects
        self.ObjectPressureThreshold = [0.010, 0.010, 0.010]  # threshold above which to trigger the rumble, psi
        self.ObjectPressureScaling = 1  # 100% pressure value for rumble.  In psi. formula is (max change in pressure - pressurethreshold)/PressureScaling

        #logging
        self.logger = None
        self.datalogger = None

        #offset
        self.offset_sensor={"num_samples":10, "time_s":0.020} #offset properties for the sensor

        #Events for determining
        self.MoveGrasperEvent = asyncio.Event()
        self.MoveGantryEvent = asyncio.Event()
        self.FreshDataEvent = asyncio.Event()
        self.GrasperReadAverage = {"Average Event":asyncio.Event(),"Number of Loops":50,"Time Delay (s)":0.005}

        #For storing position and pressure variables
        self.jawPressure = None
        self.ClosurePressure = None
        self.curPos = None #will be a Point (x,y,z) in meters

        #button and axes values
        self.buttonVal = None
        self.AxesPos = None

        #For datalogging
        self.startDatalog = asyncio.Event()
        self.finishDatalog = asyncio.Event()
        self.datalog_header = False #set to true when you've written the header for the 1st time
        self.time_0 = 0 #time for the start of the datalog

        #For SNS
        self.max_z_height = -0.184
        self.SNS_target_pos_m = [-0.19,-0.19,-0.184] #original -0.19,-0.24,-0.184
        self.SNS_object_pos_m = [-0.0095875,0.012362,-0.18464] #[0,0,-0.184]
        self.ContactThreshold = {"Pressure Threshold (psi)":[0.010,0.020,0.029], "Pressure Scaling":[1,1,1]}
        self.ContactThreshold_Rigid = {"Force Threshold (N)":0.22, "Force Scaling":2}
        self.maxJawChangeInRadius_mm = 25 #20 mm max jaw change in radius #TODO: need to change to 10 for rigid grasper, 30 mm for soft grasper
        self.SNS_BypassForceFeedback = True
        self.SNS_grasper_tc_s = 0 #time constant for the closure muscle of the grasper

        #For Calibration of pressure-Radius relationship
        self.pressure_radius_parameters = {"min pressure (psi)": 0, "max pressure (psi)": 12.0,
                                           "step pressure (psi)": 0.1, "stabilization time (s)": 5,
                                           "Pressure Capture Event": asyncio.Event(),
                                           "Pressure Actuate Event": asyncio.Event(),
                                           "Picture Capture Event" : asyncio.Event(),
                                           "Pressure Datalog Event" : asyncio.Event(),
                                           "Directory": None,
                                           "logger": None,
                                           "Video Writer": None}  # pressure capture event is to indicate object is stable, pressure actuate event is to indicate to the read_move syntax that it can move

        self.pressure_state = {"Commanded pressure (psi)": 0,
                               "Measured Pressure closure (psi)": 0,
                               "Measured Pressure Jaw 1 (psi)": 0,
                               "Measured Pressure Jaw 2 (psi)": 0,
                               "Measured Pressure Jaw 3 (psi)": 0,
                               "Time Stamp": None,
                               "Pressure Capture Event": False}

        self.video_stream = None #initialize later
        self.use_video_stream = False #set to true if you want to use the video stream, else false

        # For Koopman
        self.kpt =  {"Directory": None,
                     "logger": None,
                     "logger_header": False,
                     "logger_controls":{"x":0,"y":0,"z":0,"Grasper_Pressure":0},
                     "Object Class":"default",
                     "Object Size (mm)":0,
                     "kpm": None,
                     "proceed":False,
                     "orig_pos":[0,0,0]}

        # For LQR

        self.LQR = {"Directory": None,
                    "logger": None,
                    "logger_header": False,
                    "logger_controls":{"x":0,"y":0,"z":0,"Grasper_Pressure":0},
                    "Object Class":"default",
                    "Object Size (mm)":0,
                    "LQR": None,
                    "proceed":False,
                    "orig_pos":[0,0,0]}



    async def zero_sensors(self):


        print("Enter the number of samples and the time period to average and zero the sensors, separated by a comma. \n"
              "Enter 'D' to use the default values of %f samples, %f seconds. \n"
              "Enter 'X' to cancel."%(self.offset_sensor["num_samples"], self.offset_sensor["time_s"]))
        s = await aioconsole.ainput()
        print(s)

        match s.upper():
            # Calibration
            case "D":
                pass

            case "X":
                return

            case _:
                # set the initial
                numsamp, time_s = [float(x) for x in s.split(",")]
                self.offset_sensor["num_samples"] = numsamp
                self.offset_sensor["time_s"] = time_s


        try:
            jawVal, ClosureVal = await self.SG.ReadSensorValues(
                number_avg=self.offset_sensor["num_samples"],
                loop_delay=self.offset_sensor["time_s"])


            ## Skip this temporarily

            self.logger.info("Using: %f samples, %f seconds. \n"
                             %(self.offset_sensor["num_samples"], self.offset_sensor["time_s"]))

            if self.grasperType == GrasperType.SoftGrasper:
                self.SG.PrevJawPress = jawVal  # initialize the jaw pressure or force

                self.logger.info("jaw pressure offset (psi):" + ",".join([str(x) for x in self.SG.PrevJawPress]))

            elif self.grasperType == GrasperType.RigidGrasper:
                self.SG.PrevJawForce = jawVal  # initialize the jaw pressure or force

                self.logger.info("jaw force offset: in N: %f" %(self.SG.PrevJawForce))


        except Exception as e:

            raise e


    def setupLogger(self):
        ##### Set up logging ####
        l_date = datetime.now().strftime("_%d_%m_%Y_%H_%M_%S")


        logger_sys = logging.getLogger(__name__)

        fname = Path(__file__).parents[3].joinpath("datalogs","Hardware"+l_date+".txt")

        fh = logging.FileHandler(fname)  # file handler
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(sys.stdout)  # stream handler
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger_sys.setLevel(logging.DEBUG)
        # add the handlers to the logger_soft
        logger_sys.addHandler(fh)
        logger_sys.addHandler(ch)
        self.logger = logger_sys

        #### Set up logging data logger
        datalogger = logging.getLogger(__name__ + "_datalogs")

        fname = Path(__file__).parents[3].joinpath("datalogs", "HardwareDatalog" + l_date + ".csv")

        fh2 = logging.FileHandler(fname)  # file handler
        fh2.setLevel(logging.DEBUG)

        ch2 = logging.StreamHandler(sys.stdout)  # stream handler
        ch2.setLevel(logging.ERROR)

        formatter2 = logging.Formatter('%(asctime)s.%(msecs)03d,%(name)s,%(levelname)s,%(message)s')

        fh2.setFormatter(formatter2)
        ch2.setFormatter(formatter2)

        datalogger.setLevel(logging.DEBUG)

        # add the handlers to the logger
        datalogger.addHandler(fh2)
        datalogger.addHandler(ch2)

        self.datalogger = datalogger


    async def HardwareInitialize(self, grasper_type = GrasperType.SoftGrasper):

        self.grasperType = grasper_type

        print("Press 'S' to use Soft Grasper, 'R' to use Rigid Grasper. Any other input defaults to Soft Grasper.")
        s = await aioconsole.ainput()
        print(s)

        match s.upper():
            # Calibration
            case "S":
                self.grasperType = GrasperType.SoftGrasper
            case "R":
                self.grasperType = GrasperType.RigidGrasper
            case _:
                self.grasperType = GrasperType.SoftGrasper

        print("Press 'H' to home system, 'X' to not home system.")
        s = await aioconsole.ainput()
        home_system = False
        print(s)

        match s.upper():
            # Calibration
            case "H":
                home_system = True
            case "X":
                home_system = False
            case _:
                home_system = False


        if home_system == False:
            self.GC = GantryController(comport="COM12",homeSystem = False, initPos=[0,0,0])#, homeSystem = False,initPos=[0,0,0]  #initialize gantry controller

        elif home_system == True:
            self.GC = GantryController(comport="COM12")  # , homeSystem = False,initPos=[0,0,0]  #initialize gantry controller

        if self.grasperType == GrasperType.SoftGrasper:
            self.SG = SoftGrasper(COM_Port='COM3', BaudRate=460800, timeout=1,
                             controllerProfile="New")  # initialize soft grasper

            self.jcSG = JC.Joy_SoftGrasper(SGa=self.SG,
                                      GantryS=self.GC)  # initialize joystick control of soft grasper and gantry controller

            self.SNS_target_pos_m[2] = self.SNS_target_pos_m[2] +0.058 #add 39 mm because the grasper is now 39 mm lower than at first to accomodate the camera


        elif self.grasperType == GrasperType.RigidGrasper:  # rigid grasper
            self.SG = RigidGrasper(data_rate = 115200, port_num = "COM7", useForceSensor = True)

            self.jcSG = JC.Joy_RigidGrasper(RGa=self.SG,
                                           GantryS=self.GC)  # initialize joystick control of rigid grasper and gantry controller

            self.ObjectPressureThreshold = [0.1]  # threshold above which to trigger the rumble, psi
            self.ObjectPressureScaling = 0.05

            #self.SNS_target_pos_m[2] = self.SNS_target_pos_m[2] -0.004 #add 21 mm because the rigid grasper jaws are closer to the base platform than the soft grasper
            self.SNS_object_pos_m[2] = self.SNS_object_pos_m[2] -0.004
            self.SNS_target_pos_m[2] = self.SNS_target_pos_m[2] +0.025 #done to try without interference at release point

            #self.SG.setPID(Kp=100, Ki=2, Kd=0) #setup the gains
            # self.SG.setPID(Kp=10000, Ki=200, Kd=0)  #for the fruit
            # #self.SG.setPID(Kp=37000,Ki = 350, Kd = 0) #for the clay make sure it closes correctly
            # self.SG.getPID()

        self.SNSc = SNScontroller()

        #setup the logger
        self.setupLogger()
        self.logger.info("Grasper Type selected: %s" % self.grasperType)

    async def Read_Move_Hardware(self):
        self.SG.ReadGrasperData() #TODO ... To fix: need to do this at least once so that there is data on the serial line, otherwise "ReadSensorValues" will fail
        self.SG.ReadGrasperData()

        await asyncio.sleep(0.5)

        # if self.grasperType == GrasperType.SoftGrasper:
        #     # set the initial
        #     jawPressure, ClosurePressure = await self.SG.ReadSensorValues(
        #         number_avg=10,
        #         loop_delay=0.020)
        #
        #     ## Skip this temporarily
        #     #self.SG.PrevJawPress = jawPressure  # initialize the jaw pressure
        #
        #     self.logger.info("Initial jaw pressure is (psi):" +",".join([str(x) for x in self.SG.PrevJawPress]))

        while True:
            if self.jcSG.ControlMode != JC.JoyConState.NORMAL: #only do constant update of position when not in joystick control mode
                pass

            curPos = self.GC.getPosition()
            await asyncio.sleep(0.001)

            if self.grasperType == GrasperType.SoftGrasper:

                if self.GrasperReadAverage["Average Event"].is_set() == True: #for the case where you want to average
                    jawPressure, ClosurePressure = await self.SG.ReadSensorValues(number_avg=self.GrasperReadAverage["Number of Loops"],
                                                                                  loop_delay=self.GrasperReadAverage["Time Delay (s)"])
                    self.GrasperReadAverage["Average Event"].clear()

                else: #default
                    jawPressure, ClosurePressure = await self.SG.ReadSensorValues(number_avg=1, loop_delay=0)


                self.SG.ChangeInPressure = jawPressure
                self.jawPressure = jawPressure
                self.ClosurePressure = ClosurePressure
                self.curPos = curPos
                await asyncio.sleep(0.001)  # nominal 50 Hz

                # set the flag indicating that fresh data has been received, then sleep, then change the flag
                self.FreshDataEvent.set()
                await asyncio.sleep(0.001)

            elif self.grasperType == GrasperType.RigidGrasper:
                if self.GrasperReadAverage["Average Event"].is_set() == True:  # for the case where you want to average
                    jawForce, ClosureDistance = await self.SG.ReadSensorValues(
                        number_avg=self.GrasperReadAverage["Number of Loops"],
                        loop_delay=self.GrasperReadAverage["Time Delay (s)"])
                    self.GrasperReadAverage["Average Event"].clear()

                else:  # default
                    jawForce, ClosureDistance = await self.SG.ReadSensorValues(number_avg=1, loop_delay=0)

                self.SG.ChangeInForce = jawForce
                self.jawPressure = jawForce
                self.ClosurePressure = ClosureDistance
                self.curPos = curPos
                await asyncio.sleep(0.001)  # nominal 50 Hz

                # set the flag indicating that fresh data has been received, then sleep, then change the flag
                self.FreshDataEvent.set()
                await asyncio.sleep(0.001)


            # elif self.jcSG.ControlMode == JC.JoyConState.NORMAL:
            #     self.curPos = Point(self.GC.PositionArray["x"][-1]/1000,
            #                         self.GC.PositionArray["y"][-1]/1000,
            #                         self.GC.PositionArray["z"][-1]/1000)
            #
            #     self.ClosurePressure = self.SG.PressureArray[self.SG.closureMuscle_idx][-1]
            #     self.jawPressure = self.SG.changeInPressure

            #Move gantry only when event is set to true
            if self.MoveGantryEvent.is_set():
                self.GC.setXYZ_Position(*self.GC.goalPos, self.GC.feedrate_mmps * 60)  # absolute move of the gantry
                self.MoveGantryEvent.clear() #reset the event

            #Actuate grasper only when event is set to true
            if self.MoveGrasperEvent.is_set():
                self.SG.MoveGrasper()
                self.MoveGrasperEvent.clear() #reset the event

            #Specifically to actuate with a pressure during pressure radius cal
            if self.pressure_radius_parameters["Pressure Actuate Event"].is_set():
                self.SG.MoveGrasper_Pressure(self.pressure_state["Commanded pressure (psi)"]) #set pressure directly, don't use commanded radius position
                self.pressure_radius_parameters["Pressure Actuate Event"].clear()

            # Set the rumble
            rumbleValue = self.calculateRumble(self.ObjectPressureThreshold, self.ObjectPressureScaling)
            self.jcSG.rumbleFeedback(rumbleValue, rumbleValue, 1000)

            #indicate to datalog function that you can log the current state
            self.startDatalog.set()
            await self.finishDatalog.wait()



            # change the flags
            self.FreshDataEvent.clear()
            self.startDatalog.clear()
            await asyncio.sleep(0.001)
            # now that all the actions have been awaited, set the fresh data flag to false


    async def Normal_Mode(self): #meant to run as a long running co-routine
        #self.jcSG.PeriodT_s = 0.1 #longer in this version
        while (True):
            #print ("Normal mode")
            [buttonVal, AxesPos] = self.jcSG.eventLoop()  # run event loop to determine what values and axes to execute
            self.jcSG.ExecuteButtonFunctions(buttonVal, AxesPos)  # execute Button functions defined by the button values and axes positions

            self.buttonVal = buttonVal
            self.AxesPos = AxesPos

            if self.jcSG.ControlMode == JC.JoyConState.NORMAL:
                posInc, feedrate_mmps = self.jcSG.calcPositionIncrement(AxesPos[0], AxesPos[
                    1])  # get the joystick position for x, y and z, corrected for the flip on the x axis
                self.GC.calculateIncrementalMove(*posInc)  # will set the GC.goalPos with the appropriate increments

                self.MoveGantryEvent.set() #Move the gantry

                self.MoveGrasperEvent.set() #Move the grasper



            #Update the hardware measurements of grasper position and soft grasper state, respecting the lock




            await asyncio.sleep(0.001)  # allow other tasks to run



    async def SNS_Mode(self): #meant to run as a long running co-routine

        # Fix logging
        # Take video of it closing and record the change in radius 3 times

        while (True):
            if self.jcSG.ControlMode == JC.JoyConState.USE_SNS:
                self.logger.debug('Inside SNS control')

                self.GrasperReadAverage["Number of Loops"] = 25#15
                self.GrasperReadAverage["Time Delay (s)"] = 0.005
                self.GrasperReadAverage["Average Event"].set()  # setup to average values before returning

                await self.FreshDataEvent.wait()  # wait for fresh data

                curPos = self.curPos
                curPos_orig = curPos
                jawPressure = self.jawPressure

                if self.SNSc.first_attempt == True:
                    # z_offset = curPos.z
                    z_offset = 0
                    # will adjust height relative to object position

                    self.logger.info("Z offset %f" % z_offset)
                    object_position_list = [0, 0, self.SNS_object_pos_m[
                        2] - z_offset]  # [0, 0, -0.185] #calculate move relative to the object position.
                    # [-0.15, -0.15, -0.190]

                    target_position_list = [self.SNS_target_pos_m[0] - self.SNS_object_pos_m[0],
                                            self.SNS_target_pos_m[1] - self.SNS_object_pos_m[1],
                                            self.SNS_target_pos_m[2] - z_offset]

                    self.SNSc.first_attempt = False


                curPos = Point(-self.SNS_object_pos_m[0] + curPos.x, -self.SNS_object_pos_m[1] + curPos.y,
                               -z_offset + curPos.z)  # get where the current position is relative to where the object is

                use_noise = False #December 4th 2025: done for noise immunity tests
                if use_noise == True:
                    noise = 0.01*np.random.randn(3) #noise amplitude is 10 mm, or 0.01 meters
                    grasperPosition = Point(curPos.x + noise[0], curPos.y + noise[1], curPos.z + noise[2])  # convert to meters

                else:
                    grasperPosition = Point(curPos.x, curPos.y, curPos.z)  # convert to meters

                self.logger.debug('Target position in m: %f %f %f' % (*target_position_list,))
                self.logger.debug('GrasperPos in m:%f %f %f' % (*grasperPosition,))
                self.logger.debug('ObjectPos in m:%f %f %f' % (*object_position_list,))



                if self.grasperType == GrasperType.SoftGrasper:
                    grasperThreshold = self.ContactThreshold["Pressure Threshold (psi)"]
                    pressureScaling = self.ContactThreshold["Pressure Scaling"]

                    if self.SG.SNS_use_Force == True: # if the scaling should apply to forces instead of pressures
                        grasperContact = self.SG.GetForceFromPosition(jawPressure,grasperThreshold) # Convert psi to newtons
                        grasperContact = np.array([x*pressureScaling[i] for (i,x) in enumerate(grasperContact)])

                    else:
                        grasperContact = [(x) * pressureScaling[i] if x >= grasperThreshold[i] else 0 for (i, x) in
                                          enumerate(jawPressure)]



                    contactNum = 1 #set this to 1 to set all three jaws to the maximum of the jaw pressure. Set to 2 to set the third jaw to the maximum of the first two.

                    if contactNum == 1:
                        grasperContact = [max(grasperContact) for i in grasperContact]

                    ## grasper contact for two jaws in contact
                    elif contactNum == 2:
                        num_grasper_contact = np.where(np.array(grasperContact)==0)[0] #check how many are not in contact


                        if len(num_grasper_contact)==1: #if you have two jaws in contact, then set the third jaw to the max contact pressure
                            grasperContact[num_grasper_contact[0]] = max(grasperContact)


                    grasperContact = GrasperContactForce(*grasperContact)
                    # if len(num_grasper_contact)== 2:
                    #     grasperContact=[max(grasperContact) for x in grasperContact] #set to max contact for all three if only one jaw is in contact

                elif self.grasperType == GrasperType.RigidGrasper:
                    grasperThreshold = self.ContactThreshold_Rigid["Force Threshold (N)"]
                    forceScaling = self.ContactThreshold_Rigid["Force Scaling"]

                    grasperContact = ((jawPressure[0]) * forceScaling if jawPressure[0] >= grasperThreshold else 0)
                    #repeat the same force for all three.

                    #print('%f, %f, %f'%(jawPressure[0], grasperContact, self.SNSc.force_threshold_gain))

                    grasperContact = GrasperContactForce(grasperContact, grasperContact, grasperContact) #grasper contact force

                # TODO: able to control rigid grasper with SNS, but it does an initial off-center move and need to fix the prompts for the open loop SNS


                # check if in open loop mode and if object has been grasped and we haven't transitioned to the release phase yet
                if (self.SNSc.ControlMode == ControlType.OPEN_LOOP and self.SNSc.object_grasped_phase == True and self.SNSc.release_started == False):
                    # for grasping set to 20 psi. For releasing, use real pressure
                    if self.grasperType == GrasperType.SoftGrasper:
                        grasperContact = GrasperContactForce(*[0, 0, 0]) if self.SG.commandedPosition[
                                                                                "ClosureDistance_mm"] < self.maxJawChangeInRadius_mm else GrasperContactForce(
                            *[20, 20, 20])  # set contact threshold based on the position #maybe need to change this to see some change in pressure at the jaws before lifting up, or adding some delay time during inflation. Need to do the same during deflation
                    elif self.grasperType == GrasperType.RigidGrasper:
                        grasperContact = GrasperContactForce(*[0, 0, 0]) if self.SG.commandedPosition["ClosureDistance_mm"]  < self.maxJawChangeInRadius_mm \
                            else GrasperContactForce(*[20, 20,20])  # set contact threshold based on the position #maybe need to change this to see some change in pressure at the jaws before lifting up, or adding some delay time during inflation. Need to do the same during deflation

                #if transition to release has begun, set contact force to all zeros,

                #forward the SNS to the next time step and get the commanded xyz and radial pos of the grasper
                commandPosition_m, JawRadialPos_m = self.SNSc.SNS_forward(grasperPos_m=grasperPosition,
                                                                     grasperContact=grasperContact,
                                                                     objectPos_m=Point(*object_position_list),
                                                                     targetPos_m=Point(*target_position_list),
                                                                     useRealGantry=False) #update SNS

                self.logger.debug("After SNS forward: %f"%JawRadialPos_m)

                # ----------------- For normal mode, i.e. manually cap the grasper opening at transition from grasp to lift_after_grasp ---------------#
                if (self.SNSc.ControlMode == ControlType.NORMAL and self.SNSc.object_grasped_phase == True):
                    if self.SNSc.lift_after_grasp_started == True:
                        self.maxJawChangeInRadius_mm = min(JawRadialPos_m*1000,self.maxJawChangeInRadius_mm)
                        self.logger.info("Lift after grasp started, change in radius limited to %f"%self.maxJawChangeInRadius_mm)
                        self.logger.info("Time constant of grasper %f"%self.SNSc.controller._inter_layer_1._params["tau"].data[3])

                if (self.SNSc.ControlMode == ControlType.NORMAL and self.SNSc.lift_after_grasp_done == False):
                    self.SNSc.controller._inter_layer_1._params["tau"].data[3] = deepcopy(self.SNS_grasper_tc_s)


                elif (self.SNSc.ControlMode == ControlType.NORMAL and self.SNSc.lift_after_grasp_done == True): #if not in grasp phase, set to 0 in order to release object more quickly
                    self.SNSc.controller._inter_layer_1._params["tau"].data[3] = 0
                    self.ContactThreshold["Pressure Scaling"] = [600,600,600] #change scaling so that
                    #print(controller._inter_layer_1._params["tau"].data[3])
                # ----------------- End normal mode modification ---------------#


                # ----------------- Force Cap Mode Modification for release -----------#
                # Doesn't work because of air leakage
                # if (self.SNSc.ControlMode == ControlType.FORCE_CAP
                #         and self.SNSc.lift_after_grasp_done == True):  # if not in grasp phase, set to 0 in order to release object more quickly
                #     self.SNSc.controller._inter_layer_1._params["tau"].data[3] = 0
                #     self.SNSc.controller.grasper_closing_speed = 0.1 #set to small value
                #     self.SNSc.zero_time_constant = True
                #     self.SNSc.force_threshold_gain = 20  # change scaling so that
                #     self.logger.info("Force cap mode, modified the closing speed release")
                # ----------------- Force Cap Mode Modification for release -----------#


                # --- report for other modes what the limit of the force is when transition begins ----
                if (self.SNSc.object_grasped_phase == True and self.SNSc.lift_after_grasp_started == True):
                    self.logger.info("Lift after grasp started, change in radius set to %f" %self.SG.commandedPosition["ClosureDistance_mm"])
                    self.logger.info("Grasper Contact:")
                    self.logger.info(grasperContact)

                # --- finish

                #print(self.SNSc.neuronset)
                self.logger.debug('Jaw radial pos in m:%f' % (JawRadialPos_m))
                self.logger.debug('Command Position: %f %f %f' % (*list(commandPosition_m),))
                # command position is absolute move in m relative to the offset.
                commandPosition_m = Point(self.SNS_object_pos_m[0] + commandPosition_m.x,
                                          self.SNS_object_pos_m[1] + commandPosition_m.y,
                                          z_offset + commandPosition_m.z)  # calculate the move relative to the object position

                self.GC.goalPos = [x * 1000 for x in list(commandPosition_m)] #set goal position

                self.logger.debug('SNS commanded goal position (mm):%f,%f,%f' % (*self.GC.goalPos,))
                self.logger.debug('SNS commanded change in radius (mm):%f' % (JawRadialPos_m * 1000))
                self.logger.debug(','.join([k + ":" + str(v) for k, v in self.SNSc.neuronset.items()]))


                if self.grasperType == GrasperType.SoftGrasper:
                    self.SG.commandedPosition["ClosureDistance_mm"] = max(self.SG.max_abs_diameter_mm - 2 * (JawRadialPos_m * 1000),
                                                                        self.maxJawChangeInRadius_mm)  # limit the radial position change to prevent overinflation

                    if self.SNS_BypassForceFeedback == True and self.SNSc.object_grasped_phase == True:
                        self.SG.commandedPosition["ClosureDistance_mm"] = self.maxJawChangeInRadius_mm

                elif self.grasperType == GrasperType.RigidGrasper:
                    #note logic for commanded position is slightly different than soft grasper
                    #soft grasper uses an incremental move from the initial position.
                    #Rigid grasper is using the absolute gripper width. So set the max distance to 85 mm
                    #and then subtract 2x the SNS command since the SNS command in an incremental increase in radius
                    #max_dist = 65 if (self.SNSc.ControlMode == ControlType.FORCE_CAP and self.SNSc.object_grasped_phase == True) else 80 #maximum distance to open the jaws
                    max_dist = 65
                    self.SG.commandedPosition["ClosureDistance_mm"] = max(max_dist - 2*(JawRadialPos_m * 1000),
                                                                           self.maxJawChangeInRadius_mm) #limit to 52.5 because for the cube it takes a long time to move from 80 to 45 with large time constant. TODO: make this more elegant

                    if self.SNS_BypassForceFeedback == True and self.SNSc.object_grasped_phase == True:
                        self.SG.commandedPosition["ClosureDistance_mm"] = self.maxJawChangeInRadius_mm

                self.logger.debug('Number of grasp attempts %i' % self.SNSc.num_grasp_attempts)


                if self.SNSc.num_grasp_attempts >= 1 or self.SNSc.lift_after_release_done == True:

                    # if (SNSc.num_grasp_attempts>=1
                    #         and
                    #         (SNSc.neuronset["move_to_grasp"]>=20 or SNSc.neuronset["move_to_pre_grasp"]>=20)): #if about to attempt a regrasp
                    #     GC.goalPos = [curPos_orig.x*1000, curPos_orig.y*1000, curPos_orig.z*1000+70] #move object up
                    #     loggerR.info('Failed grasp, exceeded number of attempts')
                    #     SNSc.lift_after_grasp_done = True #set to true to trigger the next statement
                    #     SG.commandedPosition["ClosureDistance_mm"] = 0
                    #
                    if self.SNS_BypassForceFeedback == True and self.SNSc.lift_after_release_done == True: #in open loop mode, dont reattempt
                        if self.grasperType == GrasperType.SoftGrasper:
                            #self.SG.commandedPosition["ClosureDistance_mm"] = 0 #when MoveGrasper wasn't doing absolute moves
                            self.SG.commandedPosition["ClosureDistance_mm"] = self.SG.max_abs_diameter_mm

                        elif self.grasperType == GrasperType.RigidGrasper:
                            self.SG.commandedPosition["ClosureDistance_mm"] = 85

                    if self.SNSc.num_grasp_attempts >= 1 and self.SNSc.motion_complete == True:
                        self.jcSG.SNS_control = False  # reset to false to give control back to the user
                        self.jcSG.ControlMode = JC.JoyConState.NORMAL
                        self.logger.info('Reset the SNS controller after motion complete')
                        self.SG.commandedPosition["ClosureDistance_mm"] = 70 #so it doesn't re-pressurize



                # if in open loop mode, need to have extra delay to allow the grasp to complete
                if (self.SNS_BypassForceFeedback == True and self.SNSc.lift_after_grasp_started == True):
                    if self.grasperType == GrasperType.SoftGrasper:
                        if self.SG.commandedPosition[
                            "ClosureDistance_mm"] <= self.maxJawChangeInRadius_mm:  # this should always be satisfied because the contact force only is set to a large value when the commanded change in radius is larger or equal to the commanded threshold
                            self.MoveGrasperEvent.set()  # set event to indicate to other function that it should actuate grasper
                            await asyncio.sleep(20)  # sleep 20 seconds to allow the grasp to complete #hopefully only triggers once

                    elif self.grasperType == GrasperType.RigidGrasper:

                        if self.SG.commandedPosition[
                            "ClosureDistance_mm"] <= self.maxJawChangeInRadius_mm:  # this should always be satisfied because the contact force only is set to a large value when the commanded change in radius is larger or equal to the commanded threshold
                            self.MoveGrasperEvent.set()  # set event to indicate to other function that it should actuate grasper
                            await asyncio.sleep(20)  # sleep 20 seconds to allow the grasp to complete #hopefully only triggers once


                if (self.SNS_BypassForceFeedback == True and self.SNSc.neuronset["release"] > 2): #to release the object
                    if self.grasperType == GrasperType.SoftGrasper:
                        self.SG.commandedPosition["ClosureDistance_mm"] = self.SG.max_abs_diameter_mm  # need to check if this is always satisfied
                    elif self.grasperType == GrasperType.RigidGrasper:
                        self.SG.commandedPosition[
                            "ClosureDistance_mm"] = 85  #
                        
                    self.MoveGrasperEvent.set() #set event to indicate to other function that it should actuate grasper
                    await asyncio.sleep(8)
                    self.logger.debug (self.SG.commandedPosition["ClosureDistance_mm"])
                    self.logger.info("In Release Phase")
                    self.SNSc.SNS_release_done = True


                #----- For SNS force cap mode, because the jaws leak air when compressed a lot, the zero position changes. To temporarily address this, just open the grasper manually
                if (self.SNSc.ControlMode == ControlType.FORCE_CAP and self.SNSc.neuronset["release"] > 2): #to release the object
                    # need to check if this is always satisfied

                    if self.grasperType == GrasperType.SoftGrasper:
                        self.SG.commandedPosition["ClosureDistance_mm"] = self.SG.max_abs_diameter_mm  # need to check if this is always satisfied
                    elif self.grasperType == GrasperType.RigidGrasper:
                        self.SG.commandedPosition[
                            "ClosureDistance_mm"] = 85  #

                    self.MoveGrasperEvent.set() #set event to indicate to other function that it should actuate grasper
                    if self.SNSc.SNS_release_done == False:
                        await asyncio.sleep(8)
                        self.SNSc.SNS_release_done = True

                    self.logger.debug (self.SG.commandedPosition["ClosureDistance_mm"])
                    self.logger.info("In Release Phase")

                    #TODO: August 2nd: gets stuck here for a while, need to figure out why




                self.MoveGrasperEvent.set()
                self.MoveGantryEvent.set()

                self.GrasperReadAverage["Average Event"].clear()










            else:
                #self.SNSc = SNScontroller() #reinitialize SNS
                self.SNSc.first_attempt = True




            await asyncio.sleep(0.002)






    async def Calibration(self):
        while True:
            if self.jcSG.ControlMode == JC.JoyConState.CALIBRATION:
                self.GrasperReadAverage["Average Event"].set()
                self.logger.info("Calibration started. Hit SL+ to stop calibration")

                #get initial position and state of grasper

                await self.FreshDataEvent.wait() #wait for fresh data


                self.logger.info ("Gantry position at start of calibration (mm): %f %f %f"%tuple([x*1000 for x in self.curPos]))
                self.logger.info ("Grasper closure muscle pressure (psi): %f"%self.ClosurePressure) #need function to go from pressure to mm
                self.logger.info ("Grasper contact pressure (psi) at start of calibration: %f, %f, %f" % tuple(self.jawPressure))

                #close grasper
                self.SG.IncrementalMove(closureIncrement_mm=self.calibrationParams["Calibration Distance (mm)"],
                                        jawIncrement_psi=[0, 0, 0])  # setup the variables

                self.MoveGrasperEvent.set()
                self.MoveGantryEvent.set()

                #allow time to settle
                await asyncio.sleep(5) #sleep for 5 seconds.  Allow other tasks to run

                #get contact pressure prior to lift
                await self.FreshDataEvent.wait()  # wait for fresh data
                self.logger.info("Contact Pressure prior to lift: %f %f %f"%tuple(self.jawPressure))


                #lift grasper
                self.GC.calculateIncrementalMove(0,0,self.calibrationParams["Grasp Lift Height (mm)"]) #move up by 20 mm
                self.MoveGantryEvent.set()
                await asyncio.sleep(15)


                #check contact
                await self.FreshDataEvent.wait()  # wait for fresh data
                grasperContact = np.all([x if x >= self.calibrationParams["Grasp Pressure Threshold (psi)"][i] else 0 for (i, x) in
                                  enumerate(self.jawPressure)]) == True #sufficient contact if any of the thresholds greater than the threshold

                self.logger.info("Contact Pressure: %f %f %f" % tuple(self.jawPressure))

                #if object not picked up, relax grasper by 1 mm, return to home then contract by 1mm to set up next loop
                if grasperContact == False:
                    #open grasper slightly
                    self.SG.IncrementalMove(closureIncrement_mm=-2,
                                            jawIncrement_psi=[0, 0, 0])  # setup the variables
                    self.MoveGrasperEvent.set()
                    await asyncio.sleep(5)

                    #lower grasper
                    self.GC.calculateIncrementalMove(0, 0, -self.calibrationParams["Grasp Lift Height (mm)"])
                    self.MoveGantryEvent.set()
                    await asyncio.sleep(15)
                    self.logger.debug("After Sleep")


                    #close grasper slightly
                    self.SG.IncrementalMove(closureIncrement_mm=2,
                                            jawIncrement_psi=[0, 0, 0])  # setup the variables
                    self.MoveGrasperEvent.set()
                    await asyncio.sleep(5)

                #if object picked up, return to position, release object, set ControlMode to normal and report the final values.
                elif grasperContact == True:
                    self.logger.info("Calibration successful")
                    self.logger.info("Gantry position at end of calibration (mm): %f %f %f" % tuple([x * 1000 for x in self.curPos]))
                    self.logger.info(
                        "Grasper closure muscle pressure at end of calibration (psi): %f" % self.ClosurePressure)  # need function to go from pressure to mm
                    self.logger.info("Grasper closure radius at end of calibration (mm): %f" % self.SG.commandedPosition[
                        "ClosureDistance_mm"])
                    self.logger.info("Grasper contact pressure (psi) at end of calibration: %f, %f, %f" % tuple(self.jawPressure))

                    #return to home
                    self.logger.info("Returning object to base ...")
                    self.GC.calculateIncrementalMove(0, 0, -self.calibrationParams["Grasp Lift Height (mm)"])
                    self.MoveGantryEvent.set()
                    await asyncio.sleep(15)

                    #open grasper
                    self.SG.commandedPosition["ClosureDistance_mm"] = 0
                    self.MoveGrasperEvent.set()

                    await asyncio.sleep(5)

                    #set variable
                    self.jcSG.ControlMode = JC.JoyConState.NORMAL #return to normal mode

                    self.GrasperReadAverage["Average Event"].clear() #go back to only reading values 1 time

                await asyncio.sleep(0.05)  # allow other tasks to run
            #print("End of Calibration loop")
            #print(self.jcSG.ControlMode)
            await asyncio.sleep(0.001)  # allow other tasks to run
        await asyncio.sleep(0.001)

    def calculateRumble(self,pressureThreshold=[0.2, 0.2, 0.2], pressureScaling=1):

        # pressureThreshold:  change in pressure threshold in psi above which to register changes in pressure
        #TODO: Need to change scaling for rigid grasper
        if self.grasperType == GrasperType.SoftGrasper:
            rumbleValue_arr = [min((x - pressureThreshold[i]) / pressureScaling, 1) if x >= pressureThreshold[i] else 0 for
                               (i, x) in
                               enumerate(self.SG.changeInPressure)]  # was divide by 1.75, changed to 1

        elif self.grasperType == GrasperType.RigidGrasper:
            rumbleValue_arr = [min((x - pressureThreshold[i]) / pressureScaling, 1) if x >= pressureThreshold[i] else 0
                               for
                               (i, x) in
                               enumerate(self.SG.changeInForce)]  # was divide by 1.75, changed to 1

        rumbleValue = max(rumbleValue_arr)
        # if the current attempt is on a breakable item, check to see if the rumble value is greater than the threshold

        idx = [k for k, x in self.ObjectVal.items() if x.status == itemStatus.inprogress]  # find the key which in progress

        try:
            if len(idx) > 0:
                idx = idx[0]
                if self.ObjectVal[idx].itemClass == itemClass.breakable and rumbleValue > self.fragileThreshold:
                    rumbleValue = 0
                    self.logger.warning('Force exceeded on item %s.  Threshold %f, rumble value %f' %
                                    (idx, fragileThreshold, rumbleValue))


        except Exception as e:
            self.logger.error(e)
            self.logger.error('Error during fragile object check')
            rumbleValue = 0

        return (rumbleValue)

    async def ReadCommandLine(self):
        print_string = "Enter C for calibration. \n" \
                       "Enter S for SNS. \n" \
                       "Enter Z to return to joystick control\n" \
                       "Enter H to move grasper home \n" \
                       "Enter T to inflate until grasper touches \n" \
                       "Enter MG to move the grasper \n" \
                       "Enter M to move the gantry \n" \
                       "Enter D to display robot state \n" \
                       "Enter PR to enter pressure radius calibration \n" \
                       "Enter K to begin Koopman Experiments \n" \
                       "Enter LQR to begin LQR Experiments\n" \
                       "Enter ZS to zero the force/pressure sensors"
        self.logger.info(print_string)
        while (True):
            s = await aioconsole.ainput()
            self.logger.info(s)


            match s.upper():

                # Calibration
                case "C":
                    self.logger.info("Calibration Event Initiated\n")
                    self.jcSG.ControlMode = JC.JoyConState.CALIBRATION

                # SNS
                case "S":
                    self.SNSc = SNScontroller() #reinitialize each time
                    self.SNSc.initialize_controller()
                    # self.SNSc.controller.reset() #skip resetting for open-loop
                    # self.SNSc.perceptor.reset()

                    self.SNSc.first_attempt = True
                    self.jcSG.ControlMode = JC.JoyConState.PREP_SNS
                    await self.Get_SNS_Input() #prompts to setup SNS
                    self.jcSG.ControlMode = JC.JoyConState.USE_SNS

                case "T":
                    self.logger.info("Touch Event Initiated\n")
                    self.jcSG.ControlMode = JC.JoyConState.TOUCH_OBJECT

                # Return to Joystick Mode
                case "Z":
                    self.jcSG.ControlMode = JC.JoyConState.NORMAL
                    self.logger.info("Setting to NORMAL mode...\n")

                case "H":
                    await(self.returnHome())

                case "M":
                    await(self.moveGantry())

                case "MG":
                    await(self.moveGrasper())

                case "D":
                    await(self.displayRobotState())

                case "PR":
                    self.jcSG.ControlMode = JC.JoyConState.PRESSURE_RADIUS_CAL

                case "K":

                    await (self.setup_koopman())
                    self.jcSG.ControlMode = JC.JoyConState.KOOPMAN

                case "LQR":

                    await(self.setup_LQR())
                    self.jcSG.ControlMode = JC.JoyConState.LQR

                case "ZS":

                    await(self.zero_sensors())


                # Default
                case _:
                    self.logger.info(print_string)



            await asyncio.sleep(0.01)

    async def returnHome(self, defaultMode = None):
        priorMode = self.jcSG.ControlMode
        self.jcSG.ControlMode = JC.JoyConState.GO_HOME
        self.GC.goalPos = [0,0,0]
        self.MoveGantryEvent.set()
        await asyncio.sleep(20) #sleep for 20 seconds while returning home
        if defaultMode is None:
            self.jcSG.ControlMode = priorMode #return to whatever mode was set before calling this function
        else:
            self.jcSG.ControlMode = defaultMode #return to mode specified

        self.logger.info ("Return home completed")

    async def moveGantry(self):
        action_input = await aioconsole.ainput(
            "Type X to enter absolute distance in mm to move the gantry to absolute position.  \n"
            "Enter T to adjust current position by offset in mm.\n")

        match action_input.upper():
            case "T":
                vals = await aioconsole.ainput(
                    "Expecting 3 values for x,y,z offset separated by comma and in mm \n")
                vals = [float(x) for x in vals.split(',')]
                pos = [0,0,0]
                pos[0] = self.GC.goalPos[0] + vals[0]
                pos[1] = self.GC.goalPos[1] + vals[1]
                pos[2] = self.GC.goalPos[2] + vals[2]
                self.GC.goalPos = pos
            case "X":
                vals = await aioconsole.ainput(
                    "Expecting 3 values for new x,y,z separated by comma and in mm \n")
                vals = [float(x) for x in vals.split(',')]
                pos = [0,0,0]
                pos[0] = vals[0]
                pos[1] = vals[1]
                pos[2] = vals[2]
                self.GC.goalPos = pos

            case _:
                self.logger.info("Neither X or T selected. Exiting. ")

        self.MoveGantryEvent.set()

        await asyncio.sleep(0.05)


    async def moveGrasper(self):

        action_input = await aioconsole.ainput(
            "Type X to enter absolute distance in mm to move the grasper's radius to. \n"
            "Enter T to adjust current radial position in mm (+ve values close the grasper, -ve values open the grasper)\n")

        match action_input.upper():

            case "T":
                vals = await aioconsole.ainput(
                    "Expecting value in mm, this will be an offset")
                self.SG.commandedPosition["ClosureDistance_mm"] = self.SG.commandedPosition["ClosureDistance_mm"] + float(vals)

            case "X":
                vals = await aioconsole.ainput(
                    "Expecting value in mm. This is the new value for change in radial position.")
                self.SG.commandedPosition["ClosureDistance_mm"] = float(vals)

            case _:
                self.logger.info("Neither X or T selected. Exiting. ")

        self.MoveGrasperEvent.set()

        await asyncio.sleep(0.05)
    async def resetGrasper(self, defaultMode = None):
        priorMode = self.jcSG.ControlMode
        self.jcSG.ControlMode = JC.JoyConState.RESET_GRASPER
        self.SG.commandedPosition["ClosureDistance_mm"] = 0
        self.MoveGrasperEvent.set()

        await asyncio.sleep(5) #sleep for 5 seconds while returning home
        if defaultMode is None:
            self.jcSG.ControlMode = priorMode #return to whatever mode was set before calling this function
        else:
            self.jcSG.ControlMode = defaultMode #return to mode specified

        self.logger.info ("Grasper reset completed")

    async def displayRobotState(self):
        priorMode = self.jcSG.ControlMode
        self.jcSG.ControlMode = JC.JoyConState.DISPLAY_DATA
        self.GrasperReadAverage["Number of Loops"] = 15
        self.GrasperReadAverage["Time Delay (s)"] = 0.005
        await self.FreshDataEvent.wait()
        self.logger.info("Gantry position(mm): %f %f %f \n" % tuple([x * 1000 for x in self.curPos]))

        if self.grasperType == GrasperType.SoftGrasper:
            CommandedPressure = self.SG.GetPressureFromPosition(self.SG.commandedPosition["ClosureDistance_mm"])
            self.logger.info("Grasper closure muscle commanded pressure(psi):%f \n"%CommandedPressure)
            self.logger.info(
                "Grasper closure muscle pressure(psi): %f \n" % self.ClosurePressure)  # need function to go from pressure to mm
            self.logger.info("Grasper closure radius(mm): %f \n" % self.SG.commandedPosition[
                "ClosureDistance_mm"])
            self.logger.info("Grasper contact pressure (psi): %f, %f, %f \n" % tuple(self.jawPressure))

        elif self.grasperType == GrasperType.RigidGrasper:
            CurrentPosition = self.ClosurePressure #TODO: This is always 0 -> Need to fix the other todo in rigid grasper
            self.logger.info("Grasper commanded position (mm): %f \n"%self.SG.commandedPosition["ClosureDistance_mm"])
            self.logger.info("Grasper current position (mm): %f \n"%CurrentPosition)
            self.logger.info("Raw load cell reading: %f \n"%self.SG.RawForceArray[0])
            self.logger.info("Load cell reading (N): %f \n"%self.jawPressure)



        self.jcSG.ControlMode = priorMode
        await asyncio.sleep(0.001)
    async def TouchObject(self):
        while True:
            if self.jcSG.ControlMode == JC.JoyConState.TOUCH_OBJECT:
                self.GrasperReadAverage["Average Event"].set() #setup to average values before returning

                #get initial position and state of grasper

                await self.FreshDataEvent.wait() #wait for fresh data


                self.logger.info ("Gantry position at start of touch object (mm): %f %f %f"%tuple([x*1000 for x in self.curPos]))
                self.logger.info ("Grasper closure muscle pressure (psi): %f"%self.ClosurePressure) #need function to go from pressure to mm
                self.logger.info ("Grasper contact pressure (psi) at start of touch object: %f, %f, %f" % tuple(self.jawPressure))

                #close grasper
                self.SG.IncrementalMove(closureIncrement_mm=self.calibrationParams["Touch Object Distance (mm)"],
                                        jawIncrement_psi=[0, 0, 0])  # setup the variables

                self.MoveGrasperEvent.set()
                self.MoveGantryEvent.set()

                #allow time to settle
                await asyncio.sleep(1) #sleep for 1 second.  Allow other tasks to run


                #check contact
                await self.FreshDataEvent.wait()  # wait for fresh data
                grasperContact = np.all([x if x >= self.calibrationParams["Touch Pressure Threshold (psi)"][i] else 0 for (i, x) in
                                  enumerate(self.jawPressure)]) == True #sufficient contact if any of the thresholds greater than the threshold

                self.logger.info("Contact Pressure: %f %f %f" % tuple(self.jawPressure))

                #
                if grasperContact == False:

                    pass

                #if object picked up, return to position, release object, set ControlMode to normal and report the final values.
                elif grasperContact == True:
                    self.logger.info("Touch object successful")
                    self.logger.info("Gantry position at end of object touch (mm): %f %f %f" % tuple([x * 1000 for x in self.curPos]))
                    self.logger.info(
                        "Grasper closure muscle pressure at end of object touch (psi): %f" % self.ClosurePressure)  # need function to go from pressure to mm
                    self.logger.info("Grasper closure radius at end of calibration (mm): %f" % self.SG.commandedPosition[
                        "ClosureDistance_mm"])
                    self.logger.info("Grasper contact pressure (psi) at end of calibration: %f, %f, %f" % tuple(self.jawPressure))

                    #set variable
                    self.jcSG.ControlMode = JC.JoyConState.NORMAL #return to normal mode

                    self.GrasperReadAverage["Average Event"].clear() #go back to only reading values 1 time

                await asyncio.sleep(0.05)  # allow other tasks to run
            #print("End of Calibration loop")
            #print(self.jcSG.ControlMode)
            await asyncio.sleep(0.001)  # allow other tasks to run
        await asyncio.sleep(0.001)




    async def PressureRadiusCalibration(self): #meant to run as a long running co-routine

        # Finish SNS
        # Add quick close grasper step
        # Fix logging
        # Take video of it closing and record the change in radius 3 times
        #
        while (True):
            if self.jcSG.ControlMode == JC.JoyConState.PRESSURE_RADIUS_CAL:
                self.logger.info("Beginning Pressure radius cal")
                if self.video_stream is None:
                    self.logger.info("Do you want to record video? Please enter Y for yes, any other button to not record.")
                    response = await aioconsole.ainput()
                    match response.upper():
                        case "Y":
                            self.initializeVideoStream()
                            self.use_video_stream = True



                        case _:
                            self.use_video_stream = False

                if self.pressure_radius_parameters["Directory"] is None:
                    l_date = datetime.now().strftime("_%d_%m_%Y_%H_%M_%S")
                    directory_path = Path(__file__).parents[3].joinpath("datalogs", "Pressure_Radius_Test" + l_date)
                    directory_path.mkdir() #make directory
                    self.pressure_radius_parameters["Directory"] = directory_path
                    directory_image_path = directory_path.joinpath("Images")
                    directory_image_path.mkdir()

                    output_video_path = directory_image_path.joinpath("video.mp4")
                    if self.use_video_stream == True:
                        self.pressure_radius_parameters["Video Writer"] = cv.VideoWriter(str(output_video_path),
                                                                                         cv.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))




                if self.pressure_radius_parameters["logger"] is None:
                    #setup the logger
                    l_date = datetime.now().strftime("_%d_%m_%Y_%H_%M_%S")
                    datalogger = logging.getLogger(__name__ + "_datalogs")

                    fname = self.pressure_radius_parameters["Directory"].joinpath("PressureRadiusDatalog" + l_date + ".csv")

                    fh2 = logging.FileHandler(fname)  # file handler
                    fh2.setLevel(logging.DEBUG)

                    ch2 = logging.StreamHandler(sys.stdout)  # stream handler
                    ch2.setLevel(logging.INFO)

                    formatter2 = logging.Formatter('%(asctime)s,%(name)s,%(levelname)s,%(message)s')

                    fh2.setFormatter(formatter2)
                    ch2.setFormatter(formatter2)

                    datalogger.setLevel(logging.DEBUG)

                    # add the handlers to the logger
                    datalogger.addHandler(fh2)
                    datalogger.addHandler(ch2)

                    # add the header
                    datalogger.info(','.join(k for k,v in self.pressure_state.items()))

                    self.pressure_radius_parameters["logger"] = datalogger

                self.logger.info("Logger set up ...") #should be the same as the one for self.pressure_radius_parameters

                #completely deflate grasper
                self.logger.info("Completely deflating grasper...")
                self.pressure_state["Commanded pressure (psi)"] = 0

                await asyncio.sleep(5)

                #increase pressure setpoint, actuate grasper and then wait
                setpoints = np.arange(self.pressure_radius_parameters["min pressure (psi)"],
                                      self.pressure_radius_parameters["max pressure (psi)"]+
                                      self.pressure_radius_parameters["step pressure (psi)"],
                                      self.pressure_radius_parameters["step pressure (psi)"])

                for pressure in setpoints:
                    self.logger.info(pressure)

                    self.pressure_state["Commanded pressure (psi)"] = pressure
                    self.pressure_radius_parameters["Pressure Actuate Event"].set()  # set pressure actuate event

                    await asyncio.sleep(self.pressure_radius_parameters["stabilization time (s)"]) #sleep to allow it to stabilize
                    #datalog and wait until events are set properly
                    self.pressure_radius_parameters["Pressure Capture Event"].set()  # set this to allow the video capture routine to know that it should label that image specially
                    await self.pressure_radius_parameters["Pressure Datalog Event"].wait() #this will be cleared, and then set again, so wait until that is done to clear it
                    self.pressure_radius_parameters["Pressure Capture Event"].clear()

                self.logger.info("Finished pressure-radius characterization")
                self.pressure_radius_parameters["Video Writer"].release()
                self.jcSG.ControlMode = JC.JoyConState.NORMAL
            await asyncio.sleep(0.001)

    async def setup_koopman(self):
        # initialize the Koopman testing structure
        if self.kpt["kpm"] is None:
            self.kpt["kpm"] = kpm()  # initialize
            self.kpt["kpm"].compute_variable_sequence()

        if self.kpt["Directory"] is None:
            l_date = datetime.now().strftime("_%d_%m_%Y_%H_%M_%S")
            directory_path = Path(__file__).parents[3].joinpath("datalogs","Koopman Testing")
            #directory_path.mkdir()  # make directory
            self.kpt["Directory"] = directory_path

        if self.kpt["logger"] is None:
            # setup the logger
            l_date = datetime.now().strftime("_%d_%m_%Y_%H_%M_%S")
            datalogger = logging.getLogger(__name__ + "_datalog_Koopman")

            fname = self.kpt["Directory"].joinpath("Koopman_Testing" + l_date + ".csv")

            fh2 = logging.FileHandler(fname)  # file handler
            fh2.setLevel(logging.DEBUG)

            ch2 = logging.StreamHandler(sys.stdout)  # stream handler
            ch2.setLevel(logging.WARNING)

            formatter2 = logging.Formatter('%(asctime)s,%(name)s,%(levelname)s,%(message)s')

            fh2.setFormatter(formatter2)
            ch2.setFormatter(formatter2)

            datalogger.setLevel(logging.DEBUG)

            # add the handlers to the logger
            datalogger.addHandler(fh2)
            datalogger.addHandler(ch2)

            # add the header
            #datalogger.info(','.join(k for k, v in self.pressure_state.items()))

            self.kpt["logger"] = datalogger

        self.logger.info(
            "Logger set up ...")

        self.logger.info("Below are the Experimental parameters....")
        self.logger.info(self.kpt["kpm"].DF)

        # Ask User to Enter Y to proceed, enter z to exit
        orig_pos = deepcopy(self.GC.goalPos)
        self.logger.info("Enter Y to proceed, enter Z to exit")

        response = await aioconsole.ainput()

        proceed = False
        match response.upper():
            case "Y":
                proceed = True

            case Z:
                self.jcSG.ControlMode = JC.JoyConState.NORMAL
                self.logger.info("Exiting Koopman...")

        self.kpt["proceed"] = proceed
        self.logger.info("Original position (mm): %f,%f,%f" % (tuple(orig_pos)))
        self.kpt["orig_pos"] = orig_pos

        # prompt to get object type
        self.logger.info("Please enter object type (rigid, elastic or soft)\n")
        class_response = await aioconsole.ainput()
        self.kpt["Object Class"] = class_response

        # prompt to get object size
        self.logger.info("Please enter object size in mm")
        response = await aioconsole.ainput()
        self.kpt["Object Size (mm)"] = float(response)

    async def setup_LQR(self):
        # initialize the LQR Testing structure
        # self.LQR = {"Directory": None,
        #             "logger": None,
        #             "logger_header": False,
        #             "logger_controls": {"x": 0, "y": 0, "z": 0, "Grasper_Pressure": 0},
        #             "Object Class": "default",
        #             "Object Size (mm)": 0,
        #             "LQR": None,
        #             "proceed": False,
        #             "orig_pos": [0, 0, 0]}
        if self.LQR["LQR"] is None:
            self.LQR["LQR"] = LQR()  # initialize

            fname = "D://Ravesh//FRR-Software-Interface//GUI//EmbeddedSystems//Controllers//LQR_output.json"

            #--- select and read LQR file ---#
            response = await aioconsole.ainput("Press Y to select the JSON file with the A, B, C and desired Traj \n")

            match response.upper():
                case "Y":
                    fname = filedialog.askopenfilename()

                case _:
                    pass
            self.logger.info("Using LQR JSON File: %s"%fname)

            self.LQR["LQR"].read_from_JSON(fname = fname) #read the A, B, C, desired_traj from the JSON file

        if self.LQR["Directory"] is None:
            l_date = datetime.now().strftime("_%d_%m_%Y_%H_%M_%S")
            directory_path = Path(__file__).parents[3].joinpath("datalogs","LQR")
            #directory_path.mkdir()  # make directory
            self.LQR["Directory"] = directory_path

        if self.LQR["logger"] is None:
            # setup the logger
            l_date = datetime.now().strftime("_%d_%m_%Y_%H_%M_%S")
            datalogger = logging.getLogger(__name__ + "_datalog_Koopman")

            fname = self.LQR["Directory"].joinpath("LQR" + l_date + ".csv")

            fh2 = logging.FileHandler(fname)  # file handler
            fh2.setLevel(logging.DEBUG)

            ch2 = logging.StreamHandler(sys.stdout)  # stream handler
            ch2.setLevel(logging.WARNING)

            formatter2 = logging.Formatter('%(asctime)s,%(name)s,%(levelname)s,%(message)s')

            fh2.setFormatter(formatter2)
            ch2.setFormatter(formatter2)

            datalogger.setLevel(logging.DEBUG)

            # add the handlers to the logger
            datalogger.addHandler(fh2)
            datalogger.addHandler(ch2)

            # add the header
            #datalogger.info(','.join(k for k, v in self.pressure_state.items()))

            self.LQR["logger"] = datalogger

        self.logger.info(
            "Logger set up ...")


        # Ask User to Enter Y to proceed, enter z to exit
        orig_pos = deepcopy(self.GC.goalPos)
        self.logger.info("Enter Y to proceed, enter Z to exit")

        response = await aioconsole.ainput()

        proceed = False
        match response.upper():
            case "Y":
                proceed = True

            case Z:
                self.jcSG.ControlMode = JC.JoyConState.NORMAL
                self.logger.info("Exiting LQR...")

        self.LQR["proceed"] = proceed
        self.logger.info("Original position (mm): %f,%f,%f" % (tuple(orig_pos)))
        self.LQR["orig_pos"] = orig_pos

        # prompt to get object type
        self.logger.info("Please enter object type (rigid, elastic or soft)\n")
        class_response = await aioconsole.ainput()
        self.LQR["Object Class"] = class_response

        # prompt to get object size
        self.logger.info("Please enter object size in mm")
        response = await aioconsole.ainput()
        self.LQR["Object Size (mm)"] = float(response)


    async def LQRTesting(self): #meant to run as a long running co-routine



        while (True):
            if self.jcSG.ControlMode == JC.JoyConState.LQR:

                ### ---- Start LQR Experiments ---- ###
                orig_pos = self.LQR["orig_pos"]
                if self.LQR["proceed"] == True:

                    #completely deflate grasper
                    self.logger.info("Completely deflating grasper...")
                    self.pressure_state["Commanded pressure (psi)"] = 0
                    self.pressure_radius_parameters["Pressure Actuate Event"].set()
                    await asyncio.sleep(5)

                    LQRv = self.LQR["LQR"]
                    N = LQRv.N_steps #get number of points
                    start_time = datetime.now().strftime('%Y:%h:%d_%H:%M:%S')
                    testnum = 0
                    ### ---- Iterate through randomly generated sequence ---- ###
                    for i in range(0,N-1):

                        #for each timestep, get the current state and log it, then execute LQR control
                        #iterate through each variable, check if it is a control or not, then actuate accordingly
                        # datalog

                        ### ---- Get new data ---- ###
                        await self.FreshDataEvent.wait()  # await fresh data event should provide the sample freq because the serial I/O is blocking

                        new_pos_mm = [x for x in orig_pos]
                        testnum = i

                        ### ---- Get commands from LQR ---- ###
                        controls = LQRv.calculate_control(np.array(self.jawPressure), i,
                                                          use_state_estimate = True, use_affine_feedforward = True) #get LQR control based on current jaw pressures
                        jaw_pressure_psi_c, x_mm_c, y_mm_c = controls

                        self.pressure_state["Commanded pressure (psi)"] = np.clip(jaw_pressure_psi_c,0,self.SG.maxClosurePressure_psi)

                        new_pos_mm[0] = x_mm_c
                        new_pos_mm[1] = y_mm_c


                        ### ---- Send commands to perturb system ---- ###
                        self.GC.goalPos = Point(*new_pos_mm)

                        self.pressure_radius_parameters["Pressure Actuate Event"].set()  # set pressure actuate event
                        self.MoveGantryEvent.set() #set move gantry event
                        #await asyncio.sleep(1/kpmv.variables["y"].samp_freq_Hz) #wait for events to register and get executed
                        await asyncio.sleep(0.0005)



                        if self.LQR["logger_header"] == False:
                            headerstr = "Datalog_time,time_delta_s, object_class, object_size, program_mode,x_mm,y_mm,z_mm,P_closure_psi,P_jaw1_psi,P_jaw2_psi,P_jaw3_psi," \
                                        "commanded_closure_pressure_psi, commanded_P_jaw1_psi, commanded_P_jaw2_psi, commanded_P_jaw3_psi," \
                                        "commanded_x_mm, commanded_y_mm,commanded_z_mm,sequence_num," \
                                        "target_P1_psi,target_P2_psi,target_P3_psi"

                            self.LQR["logger"].info(headerstr)

                            self.LQR["logger_header"] = True

                        ### ---- Datalog events ---- ###
                        datastr = start_time + ","+ str(time.time() - self.time_0) + "," + self.LQR["Object Class"] + "," + str(self.LQR["Object Size (mm)"])
                        datastr = datastr + "," + "%s,%f,%f,%f" % (
                            str(self.jcSG.ControlMode.name), self.curPos[0] * 1000, self.curPos[1] * 1000,
                            self.curPos[2] * 1000)
                        datastr = datastr + "," + "%f,%f,%f,%f,%f,%f,%f,%f" % (
                            self.ClosurePressure, *self.jawPressure,
                            self.pressure_state["Commanded pressure (psi)"],
                            self.SG.commandedPosition["Jaw1_psi"], self.SG.commandedPosition["Jaw2_psi"],
                            self.SG.commandedPosition["Jaw3_psi"])

                        datastr = datastr + "," + ",".join([str(x) for x in self.GC.goalPos])
                        datastr = datastr + "," + str(testnum)
                        datastr = datastr + "," + ",".join([str(x) for x in LQRv.target_trajectory[:,i]])


                        self.LQR["logger"].info(datastr)

                        # sleep and set pressure datalog event
                        await asyncio.sleep(0.001)


                    ### ---- Finish For loop ---- ###
                    self.logger.info("Finished LQR")
                    self.jcSG.ControlMode = JC.JoyConState.NORMAL
            await asyncio.sleep(0.001)

    async def KoopmanTesting(self): #meant to run as a long running co-routine

        # Prompt User to hit X to proceed to Koopman. Hit Z at any time to quit.

        while (True):
            if self.jcSG.ControlMode == JC.JoyConState.KOOPMAN:

                ### ---- Start Koopman Experiments ---- ###
                orig_pos = self.kpt["orig_pos"]
                if self.kpt["proceed"] == True:

                    #completely deflate grasper
                    self.logger.info("Completely deflating grasper...")
                    self.pressure_state["Commanded pressure (psi)"] = 0
                    self.pressure_radius_parameters["Pressure Actuate Event"].set()
                    await asyncio.sleep(5)

                    kpmv = self.kpt["kpm"]
                    num_points = np.size(kpmv.variables["y"].random_sequence) #get number of points
                    start_time = datetime.now().strftime('%Y:%h:%d_%H:%M:%S')
                    testnum = 0
                    ### ---- Iterate through randomly generated sequence ---- ###
                    for i in range(0,num_points):

                        #iterate through each variable, check if it is a control or not, then actuate accordingly
                        new_pos_mm = [x for x in orig_pos]
                        for k,v in kpmv.variables.items():
                            if v.var_type == variable_type.CONTROL:
                                if v.var_name == "x":
                                    new_pos_mm[0] = new_pos_mm[0] + v.random_sequence[i]

                                elif v.var_name == "y":
                                    new_pos_mm[1] = new_pos_mm[1] + v.random_sequence[i]

                                elif v.var_name == "z":
                                    new_pos_mm[2] = new_pos_mm[2] + v.random_sequence[i]

                                elif v.var_name == "Grasper_Pressure":
                                    self.pressure_state["Commanded pressure (psi)"] = v.random_sequence[i]
                                    testnum = v.sequence_index[i]
                                else:
                                    pass
                        ### ---- Send commands to perturb system ---- ###
                        self.GC.goalPos = Point(*new_pos_mm)

                        self.pressure_radius_parameters["Pressure Actuate Event"].set()  # set pressure actuate event
                        self.MoveGantryEvent.set() #set move gantry event
                        #await asyncio.sleep(1/kpmv.variables["y"].samp_freq_Hz) #wait for events to register and get executed
                        await asyncio.sleep(0.0005)
                        #datalog
                        await self.FreshDataEvent.wait() #await fresh data event should provide the sample freq because the serial I/O is blocking

                        if self.kpt["logger_header"] == False:
                            headerstr = "Datalog_time,time_delta_s, object_class, object_size, program_mode,x_mm,y_mm,z_mm,P_closure_psi,P_jaw1_psi,P_jaw2_psi,P_jaw3_psi," \
                                        "commanded_closure_pressure_psi, commanded_P_jaw1_psi, commanded_P_jaw2_psi, commanded_P_jaw3_psi," \
                                        "commanded_x_mm, commanded_y_mm,commanded_z_mm,sequence_num"

                            self.kpt["logger"].info(headerstr)

                            self.kpt["logger_header"] = True

                        ### ---- Datalog events ---- ###
                        datastr = start_time + ","+ str(time.time() - self.time_0) + "," + self.kpt["Object Class"] + "," + str(self.kpt["Object Size (mm)"])
                        datastr = datastr + "," + "%s,%f,%f,%f" % (
                            str(self.jcSG.ControlMode.name), self.curPos[0] * 1000, self.curPos[1] * 1000,
                            self.curPos[2] * 1000)
                        datastr = datastr + "," + "%f,%f,%f,%f,%f,%f,%f,%f" % (
                            self.ClosurePressure, *self.jawPressure,
                            self.pressure_state["Commanded pressure (psi)"],
                            self.SG.commandedPosition["Jaw1_psi"], self.SG.commandedPosition["Jaw2_psi"],
                            self.SG.commandedPosition["Jaw3_psi"])

                        datastr = datastr + "," + ",".join([str(x) for x in self.GC.goalPos])
                        datastr = datastr + "," + str(testnum)


                        self.kpt["logger"].info(datastr)

                        # sleep and set pressure datalog event
                        await asyncio.sleep(0.001)


                    ### ---- Finish For loop ---- ###
                    self.logger.info("Finished Koopman")
                    self.jcSG.ControlMode = JC.JoyConState.NORMAL
            await asyncio.sleep(0.001)


    async def datalog(self): #meant to run as a long running coroutine
        while True:
            if self.jcSG.ControlMode == JC.JoyConState.PRESSURE_RADIUS_CAL and self.pressure_radius_parameters["logger"] is not None:
                await self.FreshDataEvent.wait()  # wait for fresh data
                self.pressure_state["Measured Pressure closure (psi)"] = self.ClosurePressure
                self.pressure_state["Measured Pressure Jaw 1 (psi)"] = self.jawPressure[0]
                self.pressure_state["Measured Pressure Jaw 2 (psi)"] = self.jawPressure[1]
                self.pressure_state["Measured Pressure Jaw 3 (psi)"] = self.jawPressure[2]
                self.pressure_state["Time Stamp"] = datetime.now().strftime('%Y-%m-%d-%H_%M_%S.%f')
                self.pressure_state["Pressure Capture Event"] = self.pressure_radius_parameters[
                    "Pressure Capture Event"].is_set()

                #Wait for video capture event because the botteneck is taking the picture
                await self.pressure_radius_parameters["Picture Capture Event"].wait()

                if self.pressure_state["Pressure Capture Event"] == False:
                    self.pressure_radius_parameters["logger"].debug(
                        ",".join(str(v) for k,v in self.pressure_state.items())) #log the values
                else: #to limit the amount of information displayed
                    self.pressure_radius_parameters["logger"].info(
                        ",".join(str(v) for k, v in self.pressure_state.items()))  # log the values
                    self.pressure_radius_parameters["Pressure Datalog Event"].set()






            #Normal datalog code
            #print("Datalog")
            if self.datalog_header == False:
                # write the header to the file

                if self.grasperType == GrasperType.SoftGrasper:
                    headerstr = "time_delta_s,program_mode,x_mm,y_mm,z_mm,P_closure_psi,P_jaw1_psi,P_jaw2_psi,P_jaw3_psi,commanded_radius_mm, commanded_P_jaw1_psi, commanded_P_jaw2_psi, commanded_P_jaw3_psi"

                elif self.grasperType == GrasperType.RigidGrasper:
                    headerstr = "time_delta_s,program_mode,x_mm,y_mm,z_mm," \
                              "current closure distance (mm), jaw force (N), current position 1, current position 2, currentDistance_mm, commandedPosition_mm, RawForceArray (count),ForceArray (N)"


                headerstr = headerstr + "," + ','.join(
                    [k for k, v in self.buttonVal.items()]) + "," + "Axes 1, Axes 2"  # for the buttons
                headerstr = headerstr + "," + ','.join(
                    [k for k, v in self.SNSc.neuronset.items()])  # for the SNS neuron states
                headerstr = headerstr + "," + ','.join(
                    [k for k, x in self.ObjectVal.items()])  # for the object states
                headerstr = headerstr + "," + "Mode"
                headerstr = headerstr + "," + ",".join(k for k,v in self.SNSc.motorneurons.items()) #for the SNS motor neuron commands



                self.time_0 = time.time()

                self.datalogger.info(headerstr)

                self.datalog_header = True


            '''
            Recall that header str is:
            headerstr = "program_mode,x_mm,y_mm,z_mm,P_closure_psi,P_jaw1_psi,P_jaw2_psi,P_jaw3_psi,commanded_radius_mm, commanded_P_jaw1_psi, commanded_P_jaw2_psi, commanded_P_jaw3_psi"
            headerstr = headerstr + "," + ','.join(
                [k for k, v in self.buttonVal.items()]) + "," + "Axes 1, Axes 2"  # for the buttons
            headerstr = headerstr + "," + ','.join([k for k, v in self.neuronset.items()])  # for the SNS neuron states
            headerstr = headerstr + "," + ','.join(
                [x.status.name for k, x in self.ObjectVal.items()])  # for the object states
            '''

            await self.startDatalog.wait()
            datastr = str(time.time()-self.time_0)
            datastr = datastr+ "," + "%s,%f,%f,%f"%(str(self.jcSG.ControlMode),self.curPos[0]*1000, self.curPos[1]*1000, self.curPos[2]*1000)

            if self.grasperType == GrasperType.SoftGrasper:
                datastr = datastr + "," + "%f,%f,%f,%f,%f,%f,%f,%f"%(self.ClosurePressure,*self.jawPressure,self.SG.commandedPosition["ClosureDistance_mm"],
                                                                self.SG.commandedPosition["Jaw1_psi"],self.SG.commandedPosition["Jaw2_psi"],self.SG.commandedPosition["Jaw3_psi"])
            elif self.grasperType == GrasperType.RigidGrasper:
                datastr = datastr + "," + "%f, %f, %f, %f, %f, %f, %f, %f"%(self.ClosurePressure,self.jawPressure[0],
                                               self.SG.CurrentPosition["1"], self.SG.CurrentPosition["2"],
                                               self.SG.CurrentDistance, self.SG.commandedPosition["ClosureDistance_mm"],
                                               self.SG.RawForceArray[0], self.SG.ForceArray[0][-1]) #closure distance (mm), jaw force (N), current position 1, current position 2, currentDistance_mm, commandedPosition["ClosureDistance_mm"], RawForceArray (count),ForceArray (N)

            datastr = datastr + "," + ','.join(
                [str(v) for k, v in self.buttonVal.items()]) + "," + ','.join([str(x) for x in self.AxesPos]) #for the joystick button and pos

            datastr = datastr + "," + ','.join([str(v) for k, v in self.SNSc.neuronset.items()]) #for the SNS neuronset

            datastr = datastr + "," + ','.join(["Object "+ str(x.status.name) for k, x in self.ObjectVal.items()]) #for the objects

            datastr = datastr + "," + str(self.jcSG.ControlMode.name)

            datastr = datastr + "," + ",".join(str(v) for k,v in self.SNSc.motorneurons.items())


            self.datalogger.info(datastr)
            self.finishDatalog.set()

            #sleep and set pressure datalog event
            await asyncio.sleep(0.001)
            self.pressure_radius_parameters["Pressure Datalog Event"].clear()
            self.finishDatalog.clear()

    async def capture_video(self):
        while True:
            if self.video_stream is not None:
                if self.jcSG.ControlMode == JC.JoyConState.PRESSURE_RADIUS_CAL:
                    ret, frame = self.video_stream.read()
                    if not ret:
                        self.logger.info("Can't receive frame (stream end?).")
                        break
                    #ts = datetime.now().strftime('%Y-%m-%d-%H_%M_%S.%f')
                    self.pressure_radius_parameters["Video Writer"].write(frame) #write frame to video
                    if self.pressure_state["Pressure Capture Event"] == True: #save images at end of settle time and before start of transition
                        ts = self.pressure_state["Time Stamp"]
                        fname = "IMG_ts_%s_P_%f_TF_%s.jpg"%(str(ts),
                                                                         self.pressure_state["Commanded pressure (psi)"],
                                                                         ["True" if self.pressure_state["Pressure Capture Event"] else "False"][0])
                        fname = self.pressure_radius_parameters["Directory"].joinpath("Images",fname)
                        status = cv.imwrite(str(fname), frame)
            self.pressure_radius_parameters["Picture Capture Event"].set()
            await asyncio.sleep(0.030)
            self.pressure_radius_parameters["Picture Capture Event"].clear()





    def initializeVideoStream(self,index = 0):
        self.video_stream = cv.VideoCapture(index)
        if not self.video_stream.isOpened():
            self.logger.error("Cannot open camera")






    async def Get_SNS_Input(self):


        SNS_type = await aioconsole.ainput(
                                           "Enter the type of SNS control.\n"
                                           "'O' for open-loop control.\n"
                                           "'C' for modified closed-loop force control.\n"
                                           "'FI' to enable force trigger based on inhibition of the jaw interneuron.\n"
                                           "'FC' to enable force trigger based on cap of the position based on transition.\n"
                                           "'MO' to enable SNS w/ modulation in open-loop.\n"
                                           "'MC' to enable SNS w/ modulation of force threshold"
                                            )

        self.logger.info("Type selected: %s"%SNS_type)


        match SNS_type.upper():

            case "O":
                self.SNS_BypassForceFeedback = True
                self.SNSc.ControlMode = ControlType.OPEN_LOOP
                self.SNSc.initialize_controller()
                await self.SNS_input_OpenLoop()

            case "C":
                self.SNS_BypassForceFeedback = False
                self.SNSc.ControlMode = ControlType.NORMAL
                self.SNSc.initialize_controller()
                await self.SNS_input_Normal()

            case "FI":
                self.SNS_BypassForceFeedback = False
                self.SNSc.ControlMode = ControlType.FORCE_INHIBIT
                self.SNSc.initialize_controller()
                await self.SNS_input_ForceInhibit()

            case "FC":
                self.SNS_BypassForceFeedback = False
                self.SNSc.ControlMode = ControlType.FORCE_CAP
                self.SNSc.initialize_controller()
                await self.SNS_input_ForceCap()

            case "MO":
                self.SNSc.ControlMode = ControlType.MODULATE_OPEN_LOOP
                await self.SNS_input_Modulate_Open_Loop()
                self.SNSc.initialize_controller()
                await self.SNS_input_force_threshold_gain()
                await self.SNS_input_Normal()

            case "MC":
                self.SNS_BypassForceFeedback = False
                self.SNSc.ControlMode = ControlType.MODULATE_FORCE_THRESHOLD
                self.SNSc.initialize_controller()
                await self.SNS_input_Modulate_Force_Threshold()
                await self.SNS_input_ForceCap()
                await self.SNS_z_time_constant()


            case _:
                self.logger.info("Using default")
                self.SNSc.initialize_controller()

        useXYZcalibration = await aioconsole.ainput(
            "Do you want to use the current position as the object location?\n"
            "Enter 'Y' for yes, 'N' to use default position, 'T' to adjust the calibration xyz value, or 'X' to enter a new value. \n")

        match useXYZcalibration.upper():

            case "Y":
                await self.FreshDataEvent.wait()
                self.SNS_object_pos_m =[self.curPos.x,self.curPos.y,self.curPos.z]



            case "N":
                self.logger.info("Using default values")

            case "T":
                vals = await aioconsole.ainput(
                    "Expecting 3 values for x,y,z offset separated by comma and in mm \n")
                vals = [float(x) for x in vals.split(',')]
                self.SNS_object_pos_m[0] = self.SNS_object_pos_m[0] + vals[0]/1000
                self.SNS_object_pos_m[1] = self.SNS_object_pos_m[1] + vals[1]/1000
                self.SNS_object_pos_m[2] = self.SNS_object_pos_m[2] + vals[2]/1000

            case "X":
                vals = await aioconsole.ainput(
                    "Expecting 3 values for new x,y,z separated by comma and in mm \n")
                vals = [float(x) for x in vals.split(',')]
                self.SNS_object_pos_m[0] = vals[0] / 1000
                self.SNS_object_pos_m[1] = vals[1] / 1000
                self.SNS_object_pos_m[2] = vals[2] / 1000

            case "Z":
                "Exiting SNS setup ... \n"
                return

            case _:
                pass


        self.logger.info("x,y,z used is (mm): %f, %f, %f" % (
            self.SNS_object_pos_m[0]*1000, self.SNS_object_pos_m[1]*1000, self.SNS_object_pos_m[2]*1000))

        returnHome = await aioconsole.ainput(
            "Enter any button to return home")

        await self.returnHome()

        self.logger.info("Beginning SNS ...")

    async def SNS_input_force_threshold_gain(self):
        force_gain_response = await aioconsole.ainput("Please enter 'Y' to enter the force threshold gain, "
                                                      "or enter any other button to use the default value of %f\n" % self.SNSc.force_threshold_gain)

        match force_gain_response.upper():

            case "Y":
                force_gain_value = await aioconsole.ainput(
                    "Please enter the force threshold gain as a float. i.e. 100 means 100 times the feedback force.\n "
                    "The force threshold = force_threshold_gain * feedback force\n")

                self.SNSc.force_threshold_gain = float(force_gain_value)

            case _:
                pass

        self.logger.info("Force Threshold gain is: %f" % self.SNSc.force_threshold_gain)
    async def SNS_input_ForceInhibit(self):
        await self.SNS_input_force_threshold_gain()

        inhibitory_gain_response = await aioconsole.ainput("Please enter 'Y' to enter the inhibitory threshold gain, "
                                                      "or enter any other button to use the default value of %f\n" % self.SNSc.inhibitory_gain)

        match inhibitory_gain_response.upper():

            case "Y":
                inhibitory_gain_value = await aioconsole.ainput(
                    "Please enter the inhibitory gain as a float.\n ")

                self.SNSc.inhibitory_gain = float(inhibitory_gain_value)

            case _:
                pass

        self.logger.info("Inhibitory gain is: %f" % self.SNSc.inhibitory_gain)

        # ----- prompt to get jaw thresholds and gains if the user wants to change them -----
        await self.SNS_input_jaw_thresholds_and_gains()

    async def SNS_input_ForceCap(self):
        await self.SNS_input_force_threshold_gain()

        if self.grasperType == GrasperType.SoftGrasper:
            f_y_n = await aioconsole.ainput("Please enter 'Y' to use Forces in Newtons for the SNS closed loop. "
                                             "Default is 'N'.")

            match f_y_n.upper():

                case "Y":
                    self.SG.SNS_use_Force = True #scaling applies to forces in N

                case _:
                    self.SG.SNS_use_Force = False #scaling applies to pressures in psi


        grasper_closing_speed_response = await aioconsole.ainput("Please enter 'Y' to enter the grasper closing time constant (s), "
                                                           "or enter any other button to use the default value of %f s\n" % self.SNSc.grasper_closing_speed)

        match grasper_closing_speed_response.upper():

            case "Y":
                inhibitory_gain_value = await aioconsole.ainput(
                    "Please enter the grasper closing time constant as a float.\n"
                    "The larger the value, the longer it will take to pick an object up")

                self.SNSc.grasper_closing_speed = float(inhibitory_gain_value)

            case _:
                pass

        self.logger.info("Grasper closing time constant is: %f" % self.SNSc.grasper_closing_speed)

        # ----- prompt to get jaw thresholds and gains if the user wants to change them -----
        await self.SNS_input_jaw_thresholds_and_gains()

    async def SNS_input_OpenLoop(self):
        # Ask it they want to adjust or enter new value for the grasper max closure calibration if it is open loop

        useGrasperCalibration = await aioconsole.ainput(
            "Do you want to use the grasper maximum change in radial position from the current state?\n "
            "Enter 'Y' for yes, 'N' to use default position, 'T' to adjust the calibration grasper value, or 'X' to enter a new value.\n")

        match useGrasperCalibration.upper():

            case "Y":
                await self.FreshDataEvent.wait()  # wait for fresh data
                self.maxJawChangeInRadius_mm = self.SG.commandedPosition["ClosureDistance_mm"]

            case "N":
                self.logger.info("Using default values")

            case "T":
                vals = await aioconsole.ainput(
                    "Expecting value in mm, this will be an offset")
                self.maxJawChangeInRadius_mm = self.maxJawChangeInRadius_mm + float(vals)

            case "X":
                vals = await aioconsole.ainput(
                    "Expecting value in mm. This is the new value for maximum change in radial position.")
                self.maxJawChangeInRadius_mm = float(vals)

            case "Z":
                "Exiting SNS setup ... \n"
                return

            case _:
                pass

        self.logger.info("Max closure (mm): %f" % self.maxJawChangeInRadius_mm)

        # Set the perceptor time constant: Added December 4th 2025
        if False:
            sensoryTau_response = await aioconsole.ainput("Enter 'Y' to change the Perceptor Tau.  "
                                                          "Enter 'N' otherwise.\n"
                                                          "The current value is: %f" % self.SNSc.modulation_sensory_tau)

            match sensoryTau_response.upper():
                case "Y":
                    sensoryTau = await aioconsole.ainput("Enter sensory tau as a float.\n")

                    self.SNSc.modulation_sensory_tau = float(sensoryTau)

                case "N":
                    pass

                case _:
                    pass

            self.SNSc.perceptor.set_tau(self.SNSc.modulation_sensory_tau)
            self.logger.info("SNS modulation sensory tau (s): %f" % self.SNSc.modulation_sensory_tau)



    async def SNS_input_jaw_thresholds_and_gains(self):

        if self.grasperType == GrasperType.SoftGrasper:
            SNS_thresholds_input = await aioconsole.ainput(
                "Please enter 'Y' to enter the pressure thresholds for the jaws\n"
                "Or, enter 'N' to use the default values of %f,%f,%f\n"
                "Pressures below this threshold are set to zero when computing feedback to the SNS\n" % (
                    tuple(self.ContactThreshold["Pressure Threshold (psi)"])))

        elif self.grasperType == GrasperType.RigidGrasper:
            SNS_thresholds_input = await aioconsole.ainput(
                "Please enter 'Y' to enter the force thresholds for the jaws\n"
                "Or, enter 'N' to use the default values of %f\n"
                "Forces below this threshold are set to zero when computing feedback to the SNS\n" % (
                    self.ContactThreshold_Rigid["Force Threshold (N)"]))

        match SNS_thresholds_input.upper():

            case "N":
                pass

            case "Y":

                if self.grasperType == GrasperType.SoftGrasper:
                    vals = await aioconsole.ainput(
                        "Please enter the pressure thresholds for each jaw, in psi, and separated by commas. \n")
                    vals = [float(x) for x in vals.split(',')]
                    self.ContactThreshold["Pressure Threshold (psi)"] = vals

                elif self.grasperType == GrasperType.RigidGrasper:
                    vals = await aioconsole.ainput(
                        "Please enter the force threshold in N \n")
                    vals = float(vals)
                    self.ContactThreshold_Rigid["Force Threshold (N)"] = vals

            case _:
                pass

        if self.grasperType == GrasperType.SoftGrasper:
            self.logger.info("Thresholds (psi): %f, %f, %f" % (tuple(self.ContactThreshold["Pressure Threshold (psi)"])))

        elif self.grasperType == GrasperType.RigidGrasper:
            self.logger.info(
                "Threshold (N): %f" % (self.ContactThreshold_Rigid["Force Threshold (N)"]))


        # ------ gain inputs -------#
        if self.grasperType == GrasperType.SoftGrasper:
            SNS_gains_input = await aioconsole.ainput(
                "Please enter 'Y' to enter the gains for the jaws\n"
                "Or, enter 'N' to use the default values of %f,%f,%f\n"
                "This is the multiplication factor on the measured pressure to transform it to a force.\n" % (
                    tuple(self.ContactThreshold["Pressure Scaling"])))

        elif self.grasperType == GrasperType.RigidGrasper:
            SNS_gains_input = await aioconsole.ainput(
                "Please enter 'Y' to enter the gain for the jaws\n"
                "Or, enter 'N' to use the default values of %f\n"
                "This is the multiplication factor on the measured pressure to transform it to a force.\n" % (
                    self.ContactThreshold_Rigid["Force Scaling"]))



        match SNS_gains_input.upper():

            case "N":
                pass

            case "Y":

                if self.grasperType == GrasperType.SoftGrasper:
                    vals = await aioconsole.ainput(
                        "Please enter the scaling for pressure for each jaw, and separated by commas. \n")
                    vals = [float(x) for x in vals.split(',')]
                    self.ContactThreshold["Pressure Scaling"] = vals

                elif self.grasperType == GrasperType.RigidGrasper:
                    vals = await aioconsole.ainput(
                        "Please enter the scaling for force for the jaw \n")
                    vals = float(vals)
                    self.ContactThreshold_Rigid["Force Scaling"] = vals
            case _:
                pass
        if self.grasperType == GrasperType.SoftGrasper:
            self.logger.info("Scaling: %f, %f, %f" % (tuple(self.ContactThreshold["Pressure Scaling"])))

        else:
            self.logger.info("Scaling: %f" % (self.ContactThreshold_Rigid["Force Scaling"]))

    async def SNS_input_Modulate_Open_Loop(self):
        maxPos = await aioconsole.ainput("Enter 'Y' to change the scaling of the max radial position.  Enter 'N' otherwise.\n")

        match maxPos.upper():
            case "Y":
                maxPos = await aioconsole.ainput("Enter the maximum scaling for the Radial Position.\n")
                self.SNSc.modulation_radial_scaling = float(maxPos)

            case "N":
                pass

            case _:
                pass

    async def SNS_input_Modulate_Force_Threshold(self):
        #Set the modulation gain:
        modGain_response = await aioconsole.ainput("Enter 'Y' to change the scaling of the force modulation gain.  "
                                          "Enter 'N' otherwise.\n"
                                          "The current value is: %f"%self.SNSc.modulation_mod_gain)

        match modGain_response.upper():
            case "Y":
                modGain = await aioconsole.ainput("Enter the modulation gain as a float.\n"
                                                  "This gain controls the strength of modulation. The maximal force threshold = force threshol x force modulation gain x original force thresholds\n")


                self.SNSc.modulation_mod_gain = float(modGain)

            case "N":
                pass

            case _:
                pass

        self.SNSc.perceptor.set_modulation_gain(self.SNSc.modulation_mod_gain)
        self.logger.info(
            "SNS modulation gain: %f" % self.SNSc.modulation_mod_gain)

        #Set the perceptor time constant:
        sensoryTau_response = await aioconsole.ainput("Enter 'Y' to change the Perceptor Tau.  "
                                                   "Enter 'N' otherwise.\n"
                                                   "The current value is: %f" % self.SNSc.modulation_sensory_tau)

        match sensoryTau_response.upper():
            case "Y":
                sensoryTau = await aioconsole.ainput("Enter sensory tau as a float.\n")

                self.SNSc.modulation_sensory_tau = float(sensoryTau)

            case "N":
                pass

            case _:
                pass

        self.SNSc.perceptor.set_tau(self.SNSc.modulation_sensory_tau)
        self.logger.info(
            "SNS modulation sensory tau (s): %f" % self.SNSc.modulation_sensory_tau)

        # Set the sensory gain on the force thresholds:
        sensoryGain_response = await aioconsole.ainput("Enter 'Y' to change the Sensory Gain. Note it is recommended to set the other gains to 1 for this mode.  "
                                                      "Enter 'N' otherwise.\n"
                                                      "The current value is: %f" % (1/self.SNSc.modulation_threshold_gain))

        match sensoryGain_response.upper():
            case "Y":
                sensoryGain = await aioconsole.ainput("Enter sensory gain as a float.\n"
                                                      "The maximum gain is equal to: force threshold gain x force modulation gain x original force thresholds\n"
                                                      "For pickup, the threshold is 12N for each of the three jaws \n"
                                                      "Note that it is recommended to set the other contact feedback gains to 1 when prompted for this mode. \n")

                self.SNSc.modulation_threshold_gain = 1/float(sensoryGain)

            case "N":
                pass

            case _:
                pass


        self.SNSc.perceptor.set_force_threshold(self.SNSc.modulation_threshold_gain)
        self.logger.info(
            "SNS sensory gain: %f" % self.SNSc.modulation_threshold_gain)

    async def SNS_z_time_constant(self):
        SNS_tc_input = await aioconsole.ainput(
            "Please enter 'Y' to change the time constant of the z axis during grasp\n"
            "Or, enter 'N' to use the default values of %f seconds \n" % (
                self.SNSc.controller._inter_layer_1._params["tau"].data[2]))

        match SNS_tc_input.upper():

            case "N":
                pass

            case "Y":
                vals = await aioconsole.ainput(
                    "Please enter the new time constant in seconds. \n")
                vals = float(vals)
                self.SNSc.controller._inter_layer_1._params["tau"].data[2] = vals
            case _:
                pass
        self.logger.info(
            "SNS z height time constant set to %f" % self.SNSc.controller._inter_layer_1._params["tau"].data[2])

    async def SNS_input_Normal(self):
        # Closed loop control

        # ----- thresholds and gains for jaws -----
        await self.SNS_input_jaw_thresholds_and_gains()

        # ----- z Time Constant -----
        await self.SNS_z_time_constant()
        # ----- grasper Time Constant -----
        SNS_tc_input = await aioconsole.ainput(
            "Please enter 'Y' to change the time constant of the grasper\n"
            "Or, enter 'N' to use the default values of %f seconds \n" % (
                self.SNSc.controller._inter_layer_1._params["tau"].data[3]))

        match SNS_tc_input.upper():

            case "N":
                pass

            case "Y":
                vals = await aioconsole.ainput(
                    "Please enter the new time constant in seconds. \n")
                vals = float(vals)
                self.SNSc.controller._inter_layer_1._params["tau"].data[3] = vals
            case _:
                pass

        self.SNS_grasper_tc_s = deepcopy(self.SNSc.controller._inter_layer_1._params["tau"].data[3])
        self.logger.info("SNS grasper time constant set to %f" % self.SNSc.controller._inter_layer_1._params["tau"].data[3])


if __name__ == '__main__':

    async def run_Program():
        IS = IntegratedSystem()
        await IS.HardwareInitialize()

        print("Finished Calibration")



        # https://stackoverflow.com/questions/53465862/python-aiohttp-into-existing-event-loop
        L = await asyncio.gather(
            IS.Normal_Mode(),
            IS.Calibration(),
            IS.SNS_Mode(),
            IS.ReadCommandLine(),
            IS.Read_Move_Hardware(),
            IS.TouchObject(),
            IS.datalog(),
            IS.KoopmanTesting(),
            IS.LQRTesting()
            #IS.PressureRadiusCalibration(),
            #IS.capture_video()
        )
        #runp = asyncio.create_task(HandleProgram())
        #await runp

        print('Before forever sleep')
        while True:
            await asyncio.sleep(1)



    try:
        #web.run_app(app)
        asyncio.run(run_Program())
    except KeyboardInterrupt:
        "Exiting Program.  Thank you."
