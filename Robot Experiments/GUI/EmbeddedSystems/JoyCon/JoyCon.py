import numpy as np
import pygame
from ..Gantry import GantryController as GC
from ..RigidGrasper import RigidGrasper as RG
from ..SoftGrasper import SoftGrasper as SG
from ..Support.Structures import Velocity
from enum import Enum

import logging
from datetime import datetime
import sys

from pathlib import Path


class JoyConState(Enum):
    NORMAL = 0 #move and actuate grasper based on joystick inputs
    USE_SNS = 1 #SNS is controlling the grasper position and amount of closure
    GO_HOME = 2 #Robot is moving to HOME
    CALIBRATION = 3 #Robot is performing calibration
    PREP_SNS = 4 #Getting input prompts for SNS
    TOUCH_OBJECT = 5 #move grasper until it just touches object
    RESET_GRASPER = 6 #deflate the grasper completely
    DISPLAY_DATA = 7 #display current state of robot
    PRESSURE_RADIUS_CAL = 8 #perform pressure radius calibration
    KOOPMAN = 9 #Perform Experiments for Koopman
    PID = 10 #Perform PID experiments
    LQR = 11 #Perform LQR experiments


class Button:

    def __init__(self,buttonNumber:int,buttonName:str, excludeButtons:tuple, priority:int, fcn:None):
        self.buttonNumber = buttonNumber
        self.buttonName = buttonName
        self.exclude = excludeButtons
        self.priority = priority #higher number means higher priority
        self.fcn = fcn




class JoyCon:

    def __init__(self):

        #setup logger
        self.loggerJoy = None
        self.setupLogger()

        self.joysticks = []
        self.JoyStick_JoyConID = None
        self.buttonMapping = {"A": Button(1,"A",(),1,self.buttonA), #arguments: Button Number, Button Name, Exclusion buttons, priority, button press function
                              "B": Button(3,"B",(),1,self.buttonB),
                              "X": Button(0,"X",(),1,self.buttonX),
                              "Y": Button(2,"Y",(),1,self.buttonY),
                              "SL": Button(9,"SL",(),1,self.buttonSL),
                              "SR": Button(10,"SR",(),1,self.buttonSR),
                              "+": Button(6,"+",(),1,self.buttonPlus),
                              "Stick In": Button(7,"Stick In",(),1,self.buttonJoystickIn),
                              "Home": Button(5,"Home",(),1,self.buttonHome),
                              "R": Button(16,"R",(),1,self.buttonR),
                              "ZR": Button(18,"ZR",(),1,self.buttonZR)} #For Joy-Con Right

        self.deadzone=[(-0.25,0.25),(-0.25,0.25)] #deadzone for x and y axes of the analog stick, respectively.

        # self.functionMapping = {"A": self.buttonA, "B": self.buttonB, "X": self.buttonX, "Y": self.buttonY,
        #                         "SL": self.buttonSL, "SR": self.buttonSR, "+": self.buttonPlus,
        #                         "Stick In": self.buttonJoystickIn(),"Home": self.buttonHome,
        #                         "R": self.buttonR, "ZR": self.buttonZR}
        self.initializeJoystick()

    def setupLogger(self):
        ##### Set up logging ####
        loggerJoy = logging.getLogger(__name__)
        fname = Path(__file__).parents[3].joinpath('datalogs', str(__name__) + datetime.now().strftime(
            "_%d_%m_%Y_%H_%M_%S") + ".txt")

        fh = logging.FileHandler(fname)  # file handler
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(sys.stdout)  # stream handler
        ch.setLevel(logging.ERROR)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        loggerJoy.setLevel(logging.DEBUG)
        # add the handlers to the loggerJoy
        loggerJoy.addHandler(fh)
        loggerJoy.addHandler(ch)
        self.loggerJoy = loggerJoy
    def initializeJoystick(self):
        pygame.init()

        # initialise the joystick module
        pygame.joystick.init()

        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEADDED:
                joy = pygame.joystick.Joystick(event.device_index)
                self.joysticks.append(joy)
                if joy.get_name() == "Nintendo Switch Joy-Con (R)":  # only look for Joy-Con (right)
                    self.JoyStick_JoyConID = len(self.joysticks)-1
            # quit program
            if event.type == pygame.QUIT:
                run = False

    def eventLoop(self):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                run = False

        for joystick in self.joysticks:

            if joystick.get_name() == "Nintendo Switch Joy-Con (R)": #only look for Joy-Con (right)

                buttonValues = {k: None for (k, v) in self.buttonMapping.items() if not isinstance(k,tuple)} #only get button presses, not multiple maps.
                for k, v in buttonValues.items():
                    buttonValues[k] = joystick.get_button(self.buttonMapping[k].buttonNumber)

                # player movement with analogue sticks
                vert_move = joystick.get_axis(0)
                horiz_move = joystick.get_axis(1)
                #ax3 = joystick.get_axis(4)
                #ax4 = joystick.get_axis(5)


                self.loggerJoy.debug(",".join(buttonValues.keys()))
                self.loggerJoy.debug(",".join([str(k) for k in buttonValues.values()]))
                self.loggerJoy.debug("Analog Stick: horizontal, vertical")
                self.loggerJoy.debug(str(horiz_move) + "," + str(vert_move))
                #print(str(ax3) + "," + str(ax4))

        return(buttonValues,[horiz_move,vert_move]) #return the button press values and the analog stick values


    def clear(self):
        pygame.event.clear()
    def sortButtonMapping(self):
        self.buttonMapping = dict(sorted(self.buttonMapping.items(),key=lambda x:x[1].priority,reverse = True))

    def ExecuteButtonFunctions(self,buttonValues,AnalogAxis):

        self.sortButtonMapping()  # should be in order from highest priority to lowest priority.

        #single buttons:
        singleButtons = {k: v for (k,v) in self.buttonMapping.items() if not isinstance(k, tuple)}

        #Process tuples first as these indicate simultaneous key presses
        multipleButtons = {k: v for (k, v) in self.buttonMapping.items() if
                           isinstance(k, tuple)}  # only get multiple button presses



        for (k,v) in multipleButtons.items(): #k is a key like "A" or "X", v  is a button class object
            buttonPressBool = np.all([buttonValues[x] for x in v.buttonName]) #Check if all the required buttons have been pressed. v.buttonNumber is a tuple with the buttons corresponding to the multi-button press.
            if buttonPressBool == True: #if the buttons corresponding to the multiple button command were pressed, then execute the function and remove the buttons from the single execution list

                func = [v.fcn for (k2,v) in multipleButtons.items() if frozenset(k2)==frozenset(k)][0] #get the function
                func() #call the function

                #remove  single buttons from the dictionary
                for k3 in v.exclude:
                    del singleButtons[k3] #remove buttons

        for (k, v) in singleButtons.items():
            if buttonValues[k] == 1:
                v.fcn()






    def rumbleFeedback(self,lowIntensity,highIntensity,duration_ms):
        self.joysticks[self.JoyStick_JoyConID].rumble(lowIntensity,highIntensity,duration_ms)



    #Button Functions
    def buttonA(self):
        pass

    def buttonB(self):
        pass

    def buttonX(self):
        pass

    def buttonY(self):
        pass

    def buttonHome(self):
        pass

    def buttonPlus(self):
        pass

    def buttonJoystickIn(self):
        pass

    def buttonSL(self):
        pass

    def buttonSR(self):
        pass

    def buttonR(self):
        pass

    def buttonZR(self):
        pass


class Joy_Gantry(JoyCon):

    def __init__(self,GantryS: GC.Gantry):
        super().__init__()
        self.Gantry = GantryS
        self.MoveVelocity_mmps = Velocity(*[0.6*x for x in self.Gantry.MaxSpeedRate_mmps])
        self.PeriodT_s = 0.055 #period over which to calculate movement of the gantry
        self.JoystickPos = [0,0,0] # joystick reading for x, y and z axes
        self.GantryIncrement_mm = [] #for x, y and z axes, respectively, in mm
        self.ControlMode = JoyConState.NORMAL #by default, set to Normal mode

        self.buttonMapping[("+", "Home")] = Button(
            (self.buttonMapping["+"].buttonNumber, self.buttonMapping["Home"].buttonNumber),
            ("+", "Home"),
            ("+", "Home"), 2, self.buttonHomePlus)  # situation where two buttons are pressed at once to open the grasper





    def getPositionIncrement(self, joystickPos=None, Velocity_mmps = None, PeriodT_s = None ):

        #calculate the increment for the three axes
        if Velocity_mmps is None:
            Velocity_mmps=self.MoveVelocity_mmps

        if PeriodT_s is None:
            PeriodT_s = self.PeriodT_s

        if joystickPos == None:
            joystickPos = self.JoystickPos

        #adjust value for whether in deadzone or not
        joyVal = [(x-self.deadzone[i][1] if x>self.deadzone[i][1] else 0)+
                  (x-self.deadzone[i][0] if x<self.deadzone[i][0] else 0)
                  for (i,x) in enumerate(joystickPos[0:2])] #only x and y are analog

        joyVal.append(joystickPos[2]) #z position is not analog, so just append directly
        self.JoystickPos[2] = 0 #reset to 0 after updating joyVal

        #get the increment in mm
        posIncrement_mm = [Velocity_mmps[i]*PeriodT_s*jV for (i,jV) in enumerate(joyVal)]

        feedrate_mmps = np.linalg.norm(posIncrement_mm)/PeriodT_s #feedrate is the total incremental distance moved divided by the period of motion


        self.GantryIncrement_mm = posIncrement_mm
        return(posIncrement_mm,feedrate_mmps)



    def buttonR(self):
        self.JoystickPos[2] = 1 #raise the gantry in z



    def buttonZR(self):
        self.JoystickPos[2] = -1 #lower the gantry in z


    def buttonHomePlus(self):

        self.Gantry.HomeGantry(initPos=self.Gantry.initPos)  # home the gantry

    def buttonHome(self):
        self.Gantry.setXYZ_Position(0,0,0)


    def calcPositionIncrement(self,joy_horiz_axis,joy_vert_axis):
        self.JoystickPos[0:2] = [-joy_horiz_axis,joy_vert_axis]  # modify x and y positions for the joystick pos.  the z-axis should be modified from buttonR and buttonZR calls.  X axis on the joystick is inverted from +ve motion on the joystick so multiply by -1
        posInc, feedrate_mmps = self.getPositionIncrement()  # get the position increment
        return (posInc,feedrate_mmps) #return all three x,y and z positions
    def MoveGantry_Incremental(self,joy_horiz_axis,joy_vert_axis):
        posInc, feedrate_mmps = self.calcPositionIncrement(joy_horiz_axis,joy_vert_axis) #modify x and y positions for the joystick pos.  the z-axis should be modified from buttonR and buttonZR calls.  X axis on the joystick is inverted from +ve motion on the joystick so multiply by -1

        self.Gantry.incrementalMove(moveSpeed_mmps=feedrate_mmps, **{"move_x_mm":posInc[0],"move_y_mm":posInc[1],"move_z_mm":posInc[2]})
        self.loggerJoy.debug("Joystick Pos:{0},{1}".format(joy_horiz_axis, joy_vert_axis))
        self.loggerJoy.debug("Position increment:{0},{1},{2}".format(posInc[0], posInc[1], posInc[2]))
        self.loggerJoy.debug("Current Gantry Position:{0},{1},{2}".format(self.Gantry.PositionArray["x"][-1],
                                                           self.Gantry.PositionArray["y"][-1],
                                                           self.Gantry.PositionArray["z"][-1]))




class Joy_RigidGrasper(Joy_Gantry):

    def __init__(self,RGa:RG.RigidGrasper, GantryS:GC.Gantry):
        super().__init__(GantryS = GantryS)
        self.grasper = RGa
        self.grasperIncrement_Count = 500
        self.grasperIncrement_mm = 10

        self.buttonMapping[("A", "X")] = Button((self.buttonMapping["A"].buttonNumber,self.buttonMapping["X"].buttonNumber),
                                                ("A","X"),
                                                ("A","X"),2,self.buttonAX) #situation where two buttons are pressed at once to close the grasper
        self.buttonMapping[("B", "Y")] = Button((self.buttonMapping["B"].buttonNumber,self.buttonMapping["Y"].buttonNumber),
                                                ("B","Y"),
                                                ("B","Y"),2,self.buttonBY) #situation where two buttons are pressed at once to open the grasper

        self.sortButtonMapping() #should be in order from highest priority to lowest priority.

        self.SNS_control = False


    def buttonA(self): #close both claws
        self.grasper.IncrementalMove(moveIncrement_mm= self.grasperIncrement_mm,
                                     action1=RG.GrasperActions.CLOSE,action2=RG.GrasperActions.CLOSE)

    def buttonB(self): #open both claws
        self.grasper.IncrementalMove(moveIncrement_mm=-self.grasperIncrement_mm,
                                     action1=RG.GrasperActions.OPEN, action2=RG.GrasperActions.OPEN)

    def buttonX(self):  # close left claw, assume position control
        self.grasper.IncrementalMove_Count(moveIncrement1=self.grasperIncrement_Count, moveIncrement2=self.grasperIncrement_Count,
                                     action1=RG.GrasperActions.STAY, action2=RG.GrasperActions.CLOSE)

    def buttonY(self): #open left claw, assume position control
        self.grasper.IncrementalMove_Count(moveIncrement1=self.grasperIncrement_Count, moveIncrement2=self.grasperIncrement_Count,
                                     action1=RG.GrasperActions.STAY, action2=RG.GrasperActions.OPEN)

    def buttonAX(self):
        self.grasper.IncrementalMove(moveIncrement_mm= self.grasperIncrement_mm,
                                     action1=RG.GrasperActions.CLOSE,action2=RG.GrasperActions.CLOSE)


    def buttonBY(self):
        self.grasper.IncrementalMove(moveIncrement_mm=self.grasperIncrement_mm,
                                 action1=RG.GrasperActions.OPEN, action2=RG.GrasperActions.OPEN)

    def buttonSR(self):
        self.SNS_control = True  # use SNS control

    def buttonSL(self):
        self.SNS_control = False  # don't use SNS control


    def buttonSR_Plus(self):
        self.SNS_control = not (self.SNS_control)

class Joy_SoftGrasper(Joy_Gantry):
    def __init__(self, SGa: SG.SoftGrasper , GantryS:GC.Gantry):
        super().__init__(GantryS = GantryS)
        self.grasper = SGa
        self.closureIncrement_mm = 0.5 #increment of position in mm
        self.jawIncrement_psi = 0.05  # increment of jaw pressure in psi
        self.SNS_control = False


        self.buttonMapping[("A", "X")] = Button(
            (self.buttonMapping["A"].buttonNumber, self.buttonMapping["X"].buttonNumber),
            ("A", "X"),
            ("A", "X"), 2, self.buttonAX)  # situation where two buttons are pressed at once to close the grasper
        self.buttonMapping[("B", "Y")] = Button(
            (self.buttonMapping["B"].buttonNumber, self.buttonMapping["Y"].buttonNumber),
            ("B", "Y"),
            ("B", "Y"), 2, self.buttonBY)  # situation where two buttons are pressed at once to open the grasper

        self.buttonMapping[("SR", "+")] = Button(
            (self.buttonMapping["SR"].buttonNumber, self.buttonMapping["+"].buttonNumber),
            ("SR", "+"),
            ("SR", "+"), 2, self.buttonSR_Plus)  # trigger calibration


        self.sortButtonMapping()  # should be in order from highest priority to lowest priority.



    def buttonA(self):  # close closure muscle
        self.grasper.IncrementalMove(closureIncrement_mm = self.closureIncrement_mm, jawIncrement_psi = [0,0,0])

    def buttonB(self):  # open closure muscle
        self.grasper.IncrementalMove(closureIncrement_mm = -self.closureIncrement_mm, jawIncrement_psi = [0,0,0])

    def buttonX(self):  # pressurize jaw
        self.grasper.IncrementalMove(closureIncrement_mm = 0, jawIncrement_psi = [self.jawIncrement_psi for x in range(0,3)])

    def buttonY(self):  # de-pressurize jaw
        self.grasper.IncrementalMove(closureIncrement_mm = 0, jawIncrement_psi = [-self.jawIncrement_psi for x in range(0,3)])

    def buttonSR(self):
        self.SNS_control = True #use SNS control
        self.ControlMode = JoyConState.USE_SNS

    def buttonSL(self):
        self.SNS_control = False #don't use SNS control
        self.ControlMode = JoyConState.NORMAL

    def buttonAX(self):
        pass


    def buttonBY(self):
        pass

    def buttonSR_Plus(self):
        self.ControlMode = JoyConState.CALIBRATION



# jc = JoyCon()
# jc.rumbleFeedback(0,1,5000)
# while(True):
#     [buttonVal,AxesPos] = jc.eventLoop()
#     time.sleep(0.01)
