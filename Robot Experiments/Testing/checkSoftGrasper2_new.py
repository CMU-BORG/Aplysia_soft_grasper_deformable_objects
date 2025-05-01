import time

from GUI.EmbeddedSystems.SoftGrasper.SoftGrasper import PortActions
from GUI.EmbeddedSystems.SoftGrasper.SoftGrasper import SoftGrasper

import logging

import sys
SG  = SoftGrasper(COM_Port = 'COM10',BaudRate=460800,timeout=1,controllerProfile="New")

logger_sys = logging.getLogger(__name__)
ch = logging.StreamHandler(sys.stdout)  # stream handler
ch.setLevel(logging.DEBUG)

while (True):

    # construct port commands and send to Teensy
    SG.PressurePorts[3].portStatus = PortActions.HOLD
    SG.PressurePorts[2].portStatus = PortActions.HOLD
    SG.PressurePorts[1].portStatus = PortActions.HOLD
    SG.PressurePorts[0].portStatus = PortActions.INFLATE_AND_MODULATE
    SG.PressurePorts[0].commandedPressure = 0

    byteList = SG.ConstructPortCommand()
    numBytes = SG.sendCommunicationArray(byteList=byteList)
    print("Bytes sent:%i" % (numBytes))

    # read serial data
    SG.readSerialData()
    SG.getJawChangePressureVals()
    if len(SG.PressureArray[0]) >= 1:
        print("Closer, Jaw 1, Jaw 2, Jaw 3:%f,%f,%f,%f"%(SG.PressureArray[0][-1],SG.PressureArray[1][-1],
                                                         SG.PressureArray[2][-1],SG.PressureArray[3][-1]))
    print('New')

    #print("BigLoop:"+str(vv))
    time.sleep(0.001)