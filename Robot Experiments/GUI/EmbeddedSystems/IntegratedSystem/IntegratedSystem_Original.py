import time
import asyncio
import aioconsole
import numpy as np

import logging
from datetime import datetime
import sys

from pathlib import Path

from enum import Enum

from GUI.EmbeddedSystems.SoftGrasper.SoftGrasper import PortActions
from GUI.EmbeddedSystems.SoftGrasper.SoftGrasper import SoftGrasper
from GUI.EmbeddedSystems.Gantry.GantryController import Gantry as GantryController
import GUI.EmbeddedSystems.JoyCon.JoyCon as JC
from GUI.EmbeddedSystems.SNS.SNScontroller import SNScontroller
from GUI.EmbeddedSystems.Support.Structures import GrasperContactForce,Point

import aioconsole

SG  = None #soft grasper
GC = None #gantry controller
jcSG = None #joystick
SNSc = None #SNS controller

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
        self.calibrationParams = {"Calibration Distance (mm)":1, "Grasp Pressure Threshold (psi)":[0.005,0.005,0.005], "Grasp Lift Height (mm)":35}


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
        self.setupLogger()

    def setupLogger(self):
        ##### Set up logging ####
        logger_sys = logging.getLogger(__name__)

        fname = Path(__file__).parents[3].joinpath('datalogs', str(__name__) + datetime.now().strftime(
            "_%d_%m_%Y_%H_%M_%S") + ".txt")

        fh = logging.FileHandler(fname)  # file handler
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(sys.stdout)  # stream handler
        ch.setLevel(logging.ERROR)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger_sys.setLevel(logging.DEBUG)
        # add the handlers to the logger_soft
        logger_sys.addHandler(fh)
        logger_sys.addHandler(ch)
        self.logger = logger_sys

    async def HardwareInitialize(self, grasper_type = GrasperType.SoftGrasper):

        self.grasperType = grasper_type

        self.GC = GantryController(comport="COM4",homeSystem = False, initPos=[0,0,0])#, homeSystem = False,initPos=[0,0,0]  #initialize gantry controller

        if self.grasperType == GrasperType.SoftGrasper:
            self.SG = SoftGrasper(COM_Port='COM7', BaudRate=460800, timeout=1,
                             controllerProfile="New")  # initialize soft grasper

            self.jcSG = JC.Joy_SoftGrasper(SGa=self.SG,
                                      GantryS=self.GC)  # initialize joystick control of soft grasper and gantry controller

        else:  # rigid grasper
            pass

        self.SNSc = SNScontroller(ModulateSNS=False)

    async def Normal_Mode(self): #meant to run as a long running co-routine
        while (True):
            #print ("Normal mode")
            [buttonVal, AxesPos] = self.jcSG.eventLoop()  # run event loop to determine what values and axes to execute
            self.jcSG.ExecuteButtonFunctions(buttonVal, AxesPos)  # execute Button functions defined by the button values and axes positions



            if self.jcSG.ControlMode == JC.JoyConState.NORMAL:
                posInc, feedrate_mmps = self.jcSG.calcPositionIncrement(AxesPos[0], AxesPos[
                    1])  # get the joystick position for x, y and z, corrected for the flip on the x axis
                self.GC.calculateIncrementalMove(*posInc)  # will set the GC.goalPos with the appropriate increments

                self.GC.goalPos = Point(self.GC.goalPos[0], self.GC.goalPos[1], self.GC.goalPos[2])

                self.GC.setXYZ_Position(*self.GC.goalPos, feedrate_mmps * 60)  # absolute move of the gantry

                self.SG.MoveGrasper()

            #Update the hardware measurements of grasper position and soft grasper state, respecting the lock
            # TODO: to be populated

            # Set the rumble
            rumbleValue = self.calculateRumble(self.ObjectPressureThreshold, self.ObjectPressureScaling)
            self.jcSG.rumbleFeedback(rumbleValue, rumbleValue, 1000)

            await asyncio.sleep(0.001)  # allow other tasks to run



    async def SNS_Mode(self): #meant to run as a long running co-routine
        while (True):
            # TODO: to be populated
            asyncio.sleep(0.001)






    async def Calibration(self):
        while True:
            if self.jcSG.ControlMode == JC.JoyConState.CALIBRATION:
                print("Calibration started. Hit SL+ to stop calibration")

                #get initial position and state of grasper
                curPos = self.GC.getPosition()  # get current position, in meters relative to offset
                initialJawPressure, initialClosurePressure = await self.SG.ReadSensorValues(number_avg = 20,loop_delay = 0.005)

                self.SG.ChangeInPressure = initialJawPressure
                print ("Gantry position at start of calibration (mm): %f %f %f"%tuple([x*1000 for x in curPos]))
                print ("Grasper closure muscle pressure (psi): %f"%initialClosurePressure) #need function to go from pressure to mm
                print ("Grasper contact pressure (psi) at start of calibration: %f, %f, %f" % tuple(initialJawPressure))

                #close grasper
                self.SG.IncrementalMove(closureIncrement_mm=self.calibrationParams["Calibration Distance (mm)"], jawIncrement_psi=[0, 0, 0]) #setup the variables
                self.SG.MoveGrasper() #close the grasper

                #allow time to settle
                await asyncio.sleep(5) #sleep for 5 seconds.  Allow other tasks to run

                #get contact pressure prior to lift
                contactPressure, contactClosurePressure = await self.SG.ReadSensorValues(number_avg = 20,loop_delay = 0.005)
                print("Contact Pressure: %f %f %f"%tuple(contactPressure))


                #lift grasper
                curPos = self.GC.getPosition()  # get current position, in meters relative to offset
                self.GC.incrementalMove(0,0,self.calibrationParams["Grasp Lift Height (mm)"],self.GC.MaxSpeedRate_mmps.z/2) #move up by 20 mm
                await asyncio.sleep(15)


                #check contact
                contactPressure, ClosurePressure = await self.SG.ReadSensorValues(number_avg = 20,loop_delay = 0.005)
                self.SG.ChangeInPressure = contactPressure
                grasperContact = np.all([x if x >= self.calibrationParams["Grasp Pressure Threshold (psi)"][i] else 0 for (i, x) in
                                  enumerate(contactPressure)]) == True #sufficient contact if any of the thresholds greater than the threshold


                #if object not picked up, relax grasper by 1 mm, return to home then contract by 1mm to set up next loop
                if grasperContact == False:
                    #open grasper slightly
                    self.SG.IncrementalMove(closureIncrement_mm=-2,
                                            jawIncrement_psi=[0, 0, 0])  # setup the variables
                    self.SG.MoveGrasper()  # open the grasper
                    await asyncio.sleep(5)

                    #lower grasper

                    self.GC.incrementalMove(0, 0, -self.calibrationParams["Grasp Lift Height (mm)"], self.GC.MaxSpeedRate_mmps.z/2)  # move up by 20 mm
                    await asyncio.sleep(15)
                    print("After Sleep")


                    #close grasper slightly
                    self.SG.IncrementalMove(closureIncrement_mm=2,
                                            jawIncrement_psi=[0, 0, 0])  # setup the variables
                    self.SG.MoveGrasper()  # open the grasper
                    await asyncio.sleep(5)

                #if object picked up, return to position, release object, set ControlMode to normal and report the final values.
                elif grasperContact == True:
                    print("Calibration successful")
                    print("Gantry position at end of calibration (mm): %f %f %f" % tuple([x * 1000 for x in curPos]))
                    print(
                        "Grasper closure muscle pressure at end of calibration (psi): %f" % ClosurePressure)  # need function to go from pressure to mm
                    print("Grasper closure radius at end of calibration (mm): %f" % self.SG.commandedPosition[
                        "ClosureDistance_mm"])
                    print("Grasper contact pressure (psi) at end of calibration: %f, %f, %f" % tuple(contactPressure))

                    #return to home
                    print("Returning object to base ...")
                    self.GC.incrementalMove(0, 0, -self.calibrationParams["Grasp Lift Height (mm)"],self.GC.MaxSpeedRate_mmps.z/2)  # move up by 20 mm
                    await asyncio.sleep(15)

                    #open grasper
                    self.SG.commandedPosition["ClosureDistance_mm"] = 0
                    self.SG.MoveGrasper()

                    await asyncio.sleep(5)

                    #set variable
                    self.jcSG.ControlMode = JC.JoyConState.NORMAL #return to normal mode

                await asyncio.sleep(0.05)  # allow other tasks to run
            #print("End of Calibration loop")
            #print(self.jcSG.ControlMode)
            await asyncio.sleep(0.001)  # allow other tasks to run
        await asyncio.sleep(0.001)

    def calculateRumble(self,pressureThreshold=[0.2, 0.2, 0.2], pressureScaling=1):
        global  fragileThreshold, loggerR
        # pressureThreshold:  change in pressure threshold in psi above which to register changes in pressure
        rumbleValue_arr = [min((x - pressureThreshold[i]) / pressureScaling, 1) if x >= pressureThreshold[i] else 0 for
                           (i, x) in
                           enumerate(self.SG.changeInPressure)]  # was divide by 1.75, changed to 1

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
        print_string = "Enter C for calibration. Enter S for SNS. Enter Z to return to joystick control"
        print(print_string)
        while (True):
            s = await aioconsole.ainput()
            print(s)


            match s.upper():

                case "C":
                    self.jcSG.ControlMode = JC.JoyConState.CALIBRATION

                case "S":
                    self.jcSG.ControlMode = JC.JoyConState.USE_SNS

                case "Z":
                    self.jcSG.ControlMode = JC.JoyConState.NORMAL

                case _:
                    print(print_string)



            await asyncio.sleep(0.01)


if __name__ == '__main__':

    async def run_Program():
        IS = IntegratedSystem()
        await IS.HardwareInitialize()

        print("Finished Calibration")

        IS.SG.readSerialData()
        IS.SG.readSerialData()

        IS.SG.ReadGrasperData()



        # https://stackoverflow.com/questions/53465862/python-aiohttp-into-existing-event-loop
        L = await asyncio.gather(
            IS.Normal_Mode(),
            IS.Calibration(),
            IS.ReadCommandLine()
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
