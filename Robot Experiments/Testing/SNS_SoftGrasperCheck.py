import time
import asyncio

from GUI.EmbeddedSystems.SoftGrasper.SoftGrasper import PortActions
from GUI.EmbeddedSystems.SoftGrasper.SoftGrasper import SoftGrasper
from GUI.EmbeddedSystems.Gantry.GantryController import Gantry as GantryController
import GUI.EmbeddedSystems.JoyCon.JoyCon as JC
from GUI.EmbeddedSystems.SNS.SNScontroller import SNScontroller
from GUI.EmbeddedSystems.Support.Structures import GrasperContactForce,Point

SG  = None #soft grasper
GC = None #gantry controller
jcSG = None #joystick
SNSc = None #SNS controller

# SG = SoftGrasper(COM_Port='COM5', BaudRate=460800, timeout=1, controllerProfile="New") #initialize soft grasper
# GC = GantryController(comport = "COM4",homeSystem = False, initPos=[0,0,0])#, homeSystem = False,initPos=[0,0,0]  #initialize gantry controller
# jcSG = JC.Joy_SoftGrasper(SGa=SG, GantryS=GC) #initialize joystick control of soft grasper and gantry controller
#

async def HardwareInitialize():
    global SG, GC, jcSG, SNSc
    SG = SoftGrasper(COM_Port='COM7', BaudRate=460800, timeout=1, controllerProfile="New") #initialize soft grasper
    GC = GantryController(comport = "COM4")#,homeSystem = False, initPos=[0,0,0])#, homeSystem = False,initPos=[0,0,0]  #initialize gantry controller
    jcSG = JC.Joy_SoftGrasper(SGa=SG, GantryS=GC) #initialize joystick control of soft grasper and gantry controller
    SNSc = SNScontroller(ModulateSNS=False)

async def program_loop():
    global SG, GC, jcSG, SNSc

    try:
        while(True):
            [buttonVal, AxesPos] = jcSG.eventLoop()  # run event loop to determine what values and axes to execute
            jcSG.ExecuteButtonFunctions(buttonVal,
                                        AxesPos)  # execute Button functions defined by the button values and axes positions

            if jcSG.SNS_control == False:
                posInc,feedrate_mmps = jcSG.calcPositionIncrement(AxesPos[0], AxesPos[1])  # get the joystick position for x, y and z, corrected for the flip on the x axis
                GC.calculateIncrementalMove(*posInc) #will set the GC.goalPos with the appropriate increments

            else: #SNS control
                object_position_list=[0,0,-0.185]
                target_position_list = [0.1,0.1,-0.185]
                curPos = GC.getPosition() #get current position, in millimeters relative to offset

                grasperPosition = Point(curPos.x,curPos.y,curPos.z) #convert to meters

                grasperThreshold = [0.03, 0.03, 0.03]
                grasperContact = [(x - pressureThreshold[i])*20 if x >= pressureThreshold[i] else 0 for (i, x) in
                               enumerate(SG.changeInPressure)]

                grasperContact = GrasperContactForce(*grasperContact)


                commandPosition_m, JawRadialPos_m = SNSc.SNS_forward(grasperPos_m=grasperPosition,
                                                                   grasperContact=grasperContact,
                                                                   objectPos_m=Point(*object_position_list),
                                                                   targetPos_m=Point(*target_position_list),useRealGantry = False)

                #command position is absolute move in m relative to the offset.
                GC.goalPos = [x*1000 for x in list(commandPosition_m)]

                SG.commandedPosition["ClosureDistance_mm"] = JawRadialPos_m*1000



            GC.setXYZ_Position(*GC.goalPos,feedrate_mmps*60)  # absolute move of the gantry

            SG.MoveGrasper()  # jcRG.executeButtonFunctions should update SGa with the the pressures, this command sends the appropriate commands to the grasper over serial




            # Rumble feedback based on pressure change
            pressureThreshold = [0.05, 0.05,
                                 0.05]  # change in pressure threshold in psi above which to register changes in pressure
            rumbleValue = [(x - pressureThreshold[i]) / 1.5 if x >= pressureThreshold[i] else 0 for (i, x) in
                           enumerate(SG.changeInPressure)]
            jcSG.rumbleFeedback(max(rumbleValue), max(rumbleValue), 1000)



            await asyncio.sleep(0.001) #allow other tasks to run
            print('ProgramLoop')


    except KeyboardInterrupt:
        print('Exiting loop')
        return






async def start_Program():
    await HardwareInitialize()
    await program_loop()
    #await sio.wait()





if __name__ == '__main__':
    try:
        asyncio.run(start_Program())
    except KeyboardInterrupt:
        "Exiting Program.  Thank you."