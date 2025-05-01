import time
from collections import deque

import numpy as np
import serial

ser = serial.Serial('COM7', 115200, timeout=1)
time.sleep(4)


wait_for_ok = True
Fval={"Jaw1":1,"Jaw2":0,"Jaw3":0,"I3":10,"I1_1":0,"I1_2":0,"I1_3":0}
JawValveLocations = {"Jaw1":8,"Jaw2":9,"Jaw3":11} #port numbers for the jaws to read pressure
NumPressSamples = 10 #if the pressure has been in this range for the last N samples, then hold current state and measure pressure
JawPressureTolerance = 0.1 #tolerance of pressure in which to hold Valve state
JawPressureDict = {"Jaw1":deque(maxlen=NumPressSamples),"Jaw2":deque(maxlen=NumPressSamples),"Jaw3":deque(maxlen=NumPressSamples)}
TriggerJawHoldState = {"Jaw1":False,"Jaw2":False,"Jaw3":False}
JawHoldState=0

useJawHold = True

while(True):
    while wait_for_ok:
        line = ser.readline()
        #print(line)
        wait_for_ok = True

        if line.decode().rstrip() == 'RD': #ready to accept pressure commands
            ser.write(("GO"+"\n").encode('utf-8')) #send GO signal

            for k,v in Fval.items():
                ser.write((str(v)+"\n").encode('utf-8'))

            if useJawHold == True:
                ser.write(int.to_bytes(JawHoldState,1,'big'))  #send the byte with the command for whether to open or close the pressure
            wait_for_ok = False  #only when ready for commands to you exit this loop

        elif line.decode().rstrip() == 'DL': #Device ready to send pressure readings
            # for k, v in Fval.items():
            #     line=ser.readline()
            #     val = line.decode().rstrip()
            #     print(k + " " + line.decode())
            line = ser.readline()
            val = line.decode().rstrip() #get pressure values

            print(line.decode())
            wait_for_ok = False  # only when ready for commands to you exit this loop

            #Add to Jaw pressure array and check if it is time to read contact pressure
            if useJawHold == True:
                press_arr=[float(x) for x in (line.decode()).split(',')[1:-1:2]]
                JawHoldState = 0
                counter=0
                for (k,v) in JawValveLocations.items():
                    cur_press = press_arr[JawValveLocations[k]] #get current pressure
                    JawPressureDict[k].append(cur_press) #add it to the queue
                    press_err = Fval[k]-np.array(JawPressureDict[k]) #get the error in the current pressure compared to the setpoint.
                    cur_press_err = Fval[k]-cur_press
                    if len(JawPressureDict[k])>=NumPressSamples & np.all(np.absolute(press_err)<JawPressureTolerance) & TriggerJawHoldState[k]==False:
                        TriggerJawHoldState[k] = True #Jaw will only hold pressure now unless it drops below the threshold

                    if cur_press_err>JawPressureTolerance: #based on definition of error, if it is greater than the tolerance that means it has violated the lower bound, and so we should allow the valves to work normally
                        TriggerJawHoldState[k] = False #Jaw will inlet or release air normally now

                    if TriggerJawHoldState[k] == True:
                        q=2
                        #print(k+" Contact Measuring ")

                    JawHoldState = JawHoldState|TriggerJawHoldState[k]<<counter
                    counter = counter+1


    wait_for_ok = True






# ser.write("Can you hear me?\n".encode('utf-8'))
# line = ser.readline()
# print(line.decode())

# fval = 335
# byteFval = bytearray(struct.pack("i",fval))
# binaryString="".join(["{0:08b}".format(bF) for bF in byteFval[::-1]])
# while(True):
#     ser.write(byteFval)
#     while True:
#         line = ser.readline()
#         print(line)
#         # print(line)
#         line_hold = line_hold + str(line, 'utf-8') + "\n"
#
#
#         if line == b'ok\n':
#
#             break

