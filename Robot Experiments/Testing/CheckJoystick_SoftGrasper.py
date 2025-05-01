import cProfile
import time

import JoyCon.JoyCon as JC
from Gantry.GantryController import Gantry as GC
from SoftGrasper.SoftGrasper import SoftGrasper as SG


def checkJoystick():
    SGa = SG(COM_Port = 'COM5',BaudRate=460800,timeout=1,controllerProfile="New")
    GCa = GC(comport = "COM4",homeSystem = False, initPos = [0,0,0])#, homeSystem = False,initPos=[0,0,0]  `````````````

    # Rigid Grasper Only
    #RGa = RG.RigidGrasper()
    #jcRG = JC.Joy_RigidGrasper(RGa, GantryS = GCa)

    # Gantry Only
    #jcRG = JC.Joy_Gantry(GantryS = GCa)

    # Soft Grasper Only

    jcRG = JC.Joy_SoftGrasper(SGa=SGa, GantryS=GCa)

    for i in range(0,1000000):
        [buttonVal,AxesPos] = jcRG.eventLoop() #run event loop to determine what values and axes to execute
        jcRG.ExecuteButtonFunctions(buttonVal, AxesPos) #execute Button functions defined by the button values and axes positions
        jcRG.MoveGantry(AxesPos[0], AxesPos[1]) #move the axes according to the axes
        SGa.MoveGrasper() #jcRG.executeButtonFunctions should update SGa with the the pressures, this command sends the appropriate commands to the grasper over serial

        # Rumble feedback based on pressure change
        pressureThreshold = [0.4, 0.4, 0.4] #change in pressure threshold in psi above which to register changes in pressure
        rumbleValue =[(x-pressureThreshold[i])/1.5 if x >= pressureThreshold[i] else 0 for (i,x) in enumerate(SGa.changeInPressure)]
        jcRG.rumbleFeedback(max(rumbleValue), max(rumbleValue), 1000)
        # for i, (k,v) in enumerate(buttonVal.items()):
        #     if v==1:
        #         jcRG.buttonMapping[k].fcn()
        time.sleep(0.001)



cProfile.run('checkJoystick()')