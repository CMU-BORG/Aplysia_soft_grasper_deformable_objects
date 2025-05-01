import cProfile
import time

import JoyCon.JoyCon as JC
from Gantry.GantryController import Gantry as GC
from SoftGrasper.SoftGrasper import SoftGrasper as SG


def checkJoystick():
    SGa = SG(COM_Port='COM5', BaudRate=115200, controllerProfile="Legacy")
    GCa = GC(comport = "COM4",homeSystem = False, initPos = [0,0,0])#, homeSystem = False,initPos=[0,0,0]  `````````````

    # Rigid Grasper Only
    #RGa = RG.RigidGrasper()
    #jcRG = JC.Joy_RigidGrasper(RGa, GantryS = GCa)

    # Gantry Only
    #jcRG = JC.Joy_Gantry(GantryS = GCa)

    # Soft Grasper Only

    jcRG = JC.Joy_SoftGrasper(SGa=SGa, GantryS=GCa)

    for i in range(0,1000000):
        [buttonVal,AxesPos] = jcRG.eventLoop()
        jcRG.ExecuteButtonFunctions(buttonVal, AxesPos)
        jcRG.MoveGantry(AxesPos[0], AxesPos[1])
        SGa.MoveGrasper() #program is expecting 4 bytes every loop so need to continually send move command

        # for i, (k,v) in enumerate(buttonVal.items()):
        #     if v==1:
        #         jcRG.buttonMapping[k].fcn()
        time.sleep(0.001)



cProfile.run('checkJoystick()')