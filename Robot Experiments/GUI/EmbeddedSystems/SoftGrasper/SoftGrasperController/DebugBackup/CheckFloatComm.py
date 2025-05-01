import struct
import time

import serial

ser = serial.Serial('COM7', 1000000, timeout=1)
time.sleep(8)


fval = 13.335
byteFval = bytearray(struct.pack("f",fval))
fval2 = 22.335
byteFval.extend(struct.pack("f",fval2))
#for float: struct.unpack("f",bytearray(struct.pack("f",333.3)))
binaryString="".join(["{0:08b}".format(bF) for bF in byteFval[::-1]])
while(True):
    ser.write(byteFval)
    line = ser.readline()
    print(line)
    line = ser.readline()
    print(line)
    line = ser.readline()
    print(line)




