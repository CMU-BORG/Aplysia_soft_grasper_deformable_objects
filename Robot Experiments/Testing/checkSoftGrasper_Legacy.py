from EmbeddedSystems.SoftGrasper import SoftGrasper

SG = SoftGrasper(COM_Port = 'COM5',BaudRate=115200,controllerProfile="Legacy")

while (True):
    #SendPressureCommand(ser,10.0)
    fval=10.1
    # byteFval = bytearray(struct.pack("f", fval))
    # ser.write(byteFval)
    PVal = SG.GetPressureFromPosition(5)
    print(PVal)
    SG.SendPressureCommand(PVal)
    SG.ReadPressureVals()
    if len(SG.PressureArray[0])>0:
        print(SG.PressureArray[0][-1])

    ChP = SG.getJawChangePressureVals()
    print(ChP)