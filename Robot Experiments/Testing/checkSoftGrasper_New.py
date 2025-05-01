import time

from EmbeddedSystems.SoftGrasper.SoftGrasper import PortActions
from EmbeddedSystems.SoftGrasper.SoftGrasper import SoftGrasper

SG  = SoftGrasper(COM_Port = 'COM10',BaudRate=460800,timeout=1,controllerProfile="New")
while (True):
    for i,(k,v) in enumerate(SG.PressurePorts.items()):
        SG.PressurePorts[k].commandedPressure=i
        SG.PressurePorts[k].PortStatus = PortActions.INFLATE_AND_STOP

    SG.PressurePorts[0].commandedPressure = 9
    SG.PressurePorts[0].PortStatus = PortActions.INFLATE_AND_MODULATE


    byteList = SG.ConstructPortCommand()
    SG.sendCommunicationArray(byteList=byteList)
    vv=SG.readSerialData()
    print("BigLoop:"+str(vv))
    time.sleep(0.001)
