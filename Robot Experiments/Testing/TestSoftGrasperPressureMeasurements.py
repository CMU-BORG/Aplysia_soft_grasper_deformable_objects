from GUI.EmbeddedSystems.SoftGrasper.SoftGrasper import SoftGrasper
from datetime import datetime
import time
import numpy as np

import logging

# root = logging.getLogger()
# root.setLevel(logging.DEBUG)

SG = SoftGrasper(COM_Port='COM7', BaudRate=460800, timeout=1, controllerProfile="New")  # initialize soft grasper

for i,hand in enumerate(SG.logger.handlers):
    hand.setLevel(logging.DEBUG)



t0 = time.time()
while True:
    deltaT = time.time() - t0
    #SG.readSerialData()
    SG.MoveGrasper()

    SGPressurevals = ["%2.3f"%(SG.PressureArray[i][-1] if len(SG.PressureArray[i]) > 0 else np.NaN) for i in range(0, SG.numPorts)]

    #print(','.join(SGPressurevals))
    print(SGPressurevals[SG.closureMuscle_idx]+","+SGPressurevals[SG.JawPos[0]]+","+SGPressurevals[SG.JawPos[1]]+","+SGPressurevals[SG.JawPos[2]])
    time.sleep(0.001)
