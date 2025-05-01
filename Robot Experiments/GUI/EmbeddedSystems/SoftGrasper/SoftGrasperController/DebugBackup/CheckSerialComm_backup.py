import struct

import serial


class SoftGrasper:

    def __init__(self,COM_Port = 'COM7',BaudRate=115200,timeout=1):
        self.ser = serial.Serial('COM7', 115200, timeout=1)
        self.Pressure
        self.WaitForJawsToInflate()
    def SendPressureCommand(self,PressureVal):
        byteFval = bytearray(struct.pack("f", PressureVal))
        self.ser.write(byteFval)
        line = self.ser.readline()

        return(line)
    def WaitForJawsToInflate(self):
        WaitForJawsToInflate = True
        self.ser.write("Ready\n".encode('utf-8'))

        while (WaitForJawsToInflate):
            line = self.ser.readline()
            if line.decode().rstrip() == 'Jaws finished!':
                WaitForJawsToInflate = False
            print(line)


ser = serial.Serial('COM7', 115200, timeout=1)

WaitForJawsToInflate = True
ser.write("Ready\n".encode('utf-8'))


while (WaitForJawsToInflate):
    line=ser.readline()
    if line.decode().rstrip() == 'Jaws finished!':
        WaitForJawsToInflate = False
    print(line)

while (True):
    #SendPressureCommand(ser,10.0)
    fval=10.1
    # byteFval = bytearray(struct.pack("f", fval))
    # ser.write(byteFval)
    line=SendPressureCommand(ser,10.0)
    print(line)

