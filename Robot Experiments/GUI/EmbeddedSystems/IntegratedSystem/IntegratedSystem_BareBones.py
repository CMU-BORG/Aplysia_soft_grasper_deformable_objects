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

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time



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
        self.start_time = time.time()

        self.grasperType = None

        #calibration parameters
        self.calibrationParams = {"Calibration Distance (mm)":0.5, "Grasp Pressure Threshold (psi)":[0.05,0.05,0.05], "Grasp Lift Height (mm)":20}


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
        self.ObjectPressureThreshold = [0.05, 0.05, 0.05]  # threshold above which to trigger the rumble, psi
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

        self.grasperType = grasper_typ

        self.GC = GantryController(comport="COM4")  # ,homeSystem = False, initPos=[0,0,0])#, homeSystem = False,initPos=[0,0,0]  #initialize gantry controller

        if self.grasperType == GrasperType.SoftGrasper:
            self.SG = SoftGrasper(COM_Port='COM7', BaudRate=460800, timeout=1,
                             controllerProfile="New")  # initialize soft grasper

            self.jcSG = JC.Joy_SoftGrasper(SGa=SG,
                                      GantryS=GC)  # initialize joystick control of soft grasper and gantry controller

        else:  # rigid grasper
            pass

        self.SNSc = SNScontroller(ModulateSNS=False)

    async def Normal_Mode(self): #meant to run as a long running co-routine
        while (True):

            print('Normal: %f seconds' % (time.time()-self.start_time))
            self.fragileThreshold = 0.6
            await asyncio.sleep(0.001)  # allow other tasks to run



    async def SNS_Mode(self): #meant to run as a long running co-routine
        while (True):
            # TODO: to be populated
            asyncio.sleep(0.001)




    async def Calibration(self):

        while True:

            if True:
                self.fragileThreshold = 0.6
                print('Calibration: %f seconds' % (time.time()-self.start_time))

            await asyncio.sleep(30)
            await asyncio.sleep(0.1)  # allow other tasks to run

    async def VideoCapture(self):
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            # Capture frame-by-frame
            tic = time.time()
            ret, frame = cap.read()
            toc = time.time()
            #print(toc - tic)
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            print('VideoCapture: %f seconds' % (time.time() - self.start_time))

            await asyncio.sleep(0.030)
            # # Our operations on the frame come here
            # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # # Display the resulting frame
            # plt.imshow(gray, cmap='gray', interpolation='bicubic')
            # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            # plt.show()
            # if cv.waitKey(1) == ord('q'):
            #     break
        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()


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


async def waiter(event):
    print('waiting for it ...')
    print(event.is_set())
    await event.wait()
    print(event.is_set())
    print('... got it!')


if __name__ == '__main__':

    async def run_Program():
        IS = IntegratedSystem()
        #IS.HardwareInitialize()

        event = asyncio.Event()





        # https://stackoverflow.com/questions/53465862/python-aiohttp-into-existing-event-loop
        L = await asyncio.gather(
            IS.VideoCapture(),
            IS.Normal_Mode(),
            IS.Calibration(),
            waiter(event)
        )


        #runp = asyncio.create_task(HandleProgram())
        #await runp

        # print('Before forever sleep')
        # while True:
        #     await asyncio.sleep(1)



    try:
        #web.run_app(app)
        asyncio.run(run_Program())
    except KeyboardInterrupt:
        "Exiting Program.  Thank you."
