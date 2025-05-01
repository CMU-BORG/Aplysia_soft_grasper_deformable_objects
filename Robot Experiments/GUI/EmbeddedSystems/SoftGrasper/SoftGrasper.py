import re
import struct
import sys
from enum import Enum

import numpy as np
import serial
import math

import asyncio

import logging
from datetime import datetime
import sys

from pathlib import Path

import time

#########################################################






class PortActions(Enum):
    HOLD = 0 #Don't open any valves, maintain current pressure
    INFLATE = 1 #Open inflation valve and close vacuum valve. This will increase pressure in the Artificial Muscle.
    VACUUM = 2 #Open vacuum valve and close inflation valve.  This will decrease pressure in the Artificial Muscle.
    START = 3 #Class has just been initialized.
    OPEN = 4 #Open both the inflation and vacuum valves.
    INFLATE_AND_STOP = 5 #When you first reach the set point pressure, stop
    INFLATE_AND_MODULATE = 6 #Continue to modulate pressure actively even after you first reach the set point pressure
    IGNORE = 7 #keep executing previous command
class PressurePort:

    def __init__(self,portNumber=0,portStatus=PortActions.HOLD, commandedPressure = 0.0, readPressure = 0.0, maxPressure = 12):
        self.portNumber = portNumber
        self.portStatus = portStatus
        self.commandedPressure = commandedPressure
        self.readPressure = readPressure
        self.maxPressure = maxPressure


class SoftGrasper:

    def __init__(self,COM_Port = 'COM7',BaudRate=115200,timeout=1,controllerProfile = "Legacy"):
        self.ser = serial.Serial(COM_Port, BaudRate, timeout=timeout)

        # setup Logger
        self.logger = None
        self.setupLogger()

        self.numPorts = 4 #number of ports available
        self.PressurePorts = {0 : PressurePort(0),
                              1 : PressurePort(1),
                              2 : PressurePort(2),
                              3 : PressurePort(3),
                              4 : PressurePort(4),
                              5 : PressurePort(5),
                              6 : PressurePort(6),
                              7 : PressurePort(7),
                              8 : PressurePort(8),
                              9 : PressurePort(9),
                              10 : PressurePort(10),
                              11 : PressurePort(11)
                              } #to hold the pressure values and the status of the ports


        self.PressureArray = [[] for x in range(0, self.numPorts)]  # to store pressure values
        self.PrevJawPress = [0,0,0]  # list to hold the original Jaw Press. Need to update outside of code
        self.JawPos = [1, 2, 3]  # position of pressure values that the jaws are at
        self.closureMuscle_idx = 0 #index for the closure muscle
        self.changeInPressure = [0, 0, 0] # change in pressure in psi for the three jaws
        self.maxClosurePressure_psi = 12 #maximum pressure for the closure muscle in psi
        self.min_abs_diameter_mm = 26 #minimum absolute diameter achievable in mm
        self.max_abs_diameter_mm = 66.16 #maximum absolute diameter achievable in mm


        #Tx-Rx Information for New Protocol
        self.startChar = ">!" #indicates start of comm
        self.endChar = "!<" #indicates end of comm
        self.messageStarted = False #true if you have received the startChar
        self.ProtocolSizeStarted = False #true if you have received the protocol type and size
        self.PayloadStarted = False # true if you have received the payload
        self.protocol_type = None #type of message received
        self.payload_size = 0 #size of the payload
        self.txData = None #data read

        #Tx-Rx Information for the modified protocol:
        self.prevBuffer = bytearray() #empty byte array that unprocessed data at the end of the buffer will be processed with

        #What form of the controller  to use: Legacy or New
        self.controllerProfile = controllerProfile

        #Variables for keeping track of commanded position
        self.commandedPosition = {"ClosureDistance_mm":self.max_abs_diameter_mm, "Jaw1_psi":0, "Jaw2_psi":0, "Jaw3_psi":0}

        #For GUI real-time control
        self.isActive = False

        #Max closure pressure set
        self.PressurePorts[self.closureMuscle_idx].maxPressure = self.maxClosurePressure_psi

        #Use force in SNS feedback if true, else use pressure if false.
        self.SNS_use_Force = False
        # #------ Initialization -----#
        # if self.controllerProfile == "Legacy":
        #     self.WaitForJawsToInflate()
        # else:
        #     self.WaitForTeensyInitialization()

    def setupLogger(self):
        ##### Set up logging ####
        logger_soft = logging.getLogger(__name__)

        fname = Path(__file__).parents[3].joinpath('datalogs', str(__name__) + datetime.now().strftime(
            "_%d_%m_%Y_%H_%M_%S") + ".txt")

        fh = logging.FileHandler(fname)  # file handler
        fh.setLevel(logging.ERROR) #set to error otherwise takes forever

        ch = logging.StreamHandler(sys.stdout)  # stream handler
        ch.setLevel(logging.ERROR)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger_soft.setLevel(logging.DEBUG)
        # add the handlers to the logger_soft
        logger_soft.addHandler(fh)
        logger_soft.addHandler(ch)
        self.logger = logger_soft

    def GetForceFromPosition(self, pressure_arr,pressureThreshold = 0 ):

        def calcForce(p):
            a = 0.2403
            b = 8.8984
            c = -0.2158
            d = -18.5218
            F = a*np.exp(b*p) + c*np.exp(d*p)
            return F

        force_arr = [calcForce(p)*(p>=pressureThreshold[i]) for [i,p] in enumerate(pressure_arr)]
        return(force_arr)



    def GetPressureFromPosition(self,position_mm,coeffs=[-0.0000001777329,
                                                         0.00004792679,
                                                         -0.005318879,
                                                         0.3107475,
                                                         -10.06705,
                                                         170.9448,
                                                         -1174.247]):
        b = coeffs
        x = position_mm
        x=np.clip(x,self.min_abs_diameter_mm,self.max_abs_diameter_mm)  #limit to maximum  mm contraction
        pressV = b[0]*(x**6) + b[1]*(x**5) + b[2]*(x**4) + b[3]*(x**3) + b[4]*(x**2) + b[5]*(x) + b[6]*1
        pressV = np.clip(pressV,0,self.maxClosurePressure_psi)  #limit pressure
        return(pressV)
    def SendPressureCommand(self,PressureVal):
        byteFval = bytearray(struct.pack("f", PressureVal))
        self.ser.write(byteFval)

    def WaitForTeensyInitialization(self):
        self.logger.info("Waiting for Arduino to initialize")

        FinishedInitialization = False
        while (not FinishedInitialization):
            msg = self.ser.readline()
            msg = msg.decode().rstrip()
            if msg == 'Teensy Ready!':
                self.logger.info("Soft Grasper Teensy Initialized!")
                FinishedInitialization = True

        return (FinishedInitialization)

    def WaitForJawsToInflate(self):
        WaitForJawsToInflate = True
        self.ser.write("Ready\n".encode('utf-8'))

        while (WaitForJawsToInflate):
            line = self.ser.readline()
            if line.decode().rstrip() == 'Jaws finished!':
                WaitForJawsToInflate = False
            self.logger.debug(line)


    def ReadPressureVals(self):
        line = self.ser.readline().decode()
        try:
            pVal = [float(x) for x in line.split(",")[1:]]
            if len(pVal)!=self.numPorts:
                self.logger.error("Not enough pressure values:")
                self.logger.error(line)
            else:
                for count,val in enumerate(pVal):
                    self.PressureArray[count].append(pVal[count])

        except Exception as e:
            self.logger.exception("Error during read")

    def getJawChangePressureVals(self):
        if (len(self.PressureArray[self.JawPos[0]])<=2):
            return([0,0,0])
        else:

            CurJawPress = [self.PressureArray[x][-1] for x in self.JawPos]
            #PrevJawPress= [self.PressureArray[x][-2] for x in self.JawPos]
            # if self.PrevJawPress is None:
            #     self.PrevJawPress=CurJawPress
            #     #self.logger.info('baseline Jaw pressure is '+str(self.PrevJawPress))

            ChangeInPressure = (np.array(CurJawPress)-np.array(self.PrevJawPress)).tolist()

            #ChangeInPressure = (np.array(CurJawPress)).tolist()
            #return(ChangeInPressure)
            return(ChangeInPressure)

    def IncrementalMove(self, closureIncrement_mm = 0, jawIncrement_psi = [0,0,0]):
        self.commandedPosition["ClosureDistance_mm"] = self.commandedPosition["ClosureDistance_mm"] - closureIncrement_mm
        self.logger.debug("Commanded Position in mm: "+str(self.commandedPosition["ClosureDistance_mm"] )) #for debug
        self.commandedPosition["Jaw1_psi"] = min(max(0,self.commandedPosition["Jaw1_psi"] + jawIncrement_psi[0]),2)
        self.commandedPosition["Jaw2_psi"] = min(max(0,self.commandedPosition["Jaw2_psi"] + jawIncrement_psi[1]),2)
        self.commandedPosition["Jaw3_psi"] = min(max(0,self.commandedPosition["Jaw3_psi"] + jawIncrement_psi[2]),2)

    def AbsoluteMove(self, closureIncrement_mm = 0, jawIncrement_psi = [0,0,0]):
        self.commandedPosition["ClosureDistance_mm"] = min(max(0, closureIncrement_mm),23)
        self.logger.debug("Commanded Position in mm: "+str(self.commandedPosition["ClosureDistance_mm"] )) #for debug
        self.commandedPosition["Jaw1_psi"] = min(max(0, jawIncrement_psi[0]),2)
        self.commandedPosition["Jaw2_psi"] = min(max(0,jawIncrement_psi[1]),2)
        self.commandedPosition["Jaw3_psi"] = min(max(0, jawIncrement_psi[2]),2)


    def MoveGrasper_DEPRECATED(self):
        PVal = self.GetPressureFromPosition(self.commandedPosition["ClosureDistance_mm"])
        self.logger.debug(PVal)
        self.SendPressureCommand(PVal)
        self.ReadPressureVals()

        if len(self.PressureArray[0]) > 0:
            self.logger.debug("Pressure val: " + str(self.PressureArray[0][-1]))


        #ChP = SG.getJawChangePressureVals()


    def MoveGrasper(self):
        PVal = self.GetPressureFromPosition(self.commandedPosition["ClosureDistance_mm"]) #get the pressure value in psi
        self.MoveGrasper_Pressure(PVal)


    def MoveGrasper_Pressure(self,PVal):
        self.logger.debug("CommandedPressure: " + str(PVal))

        self.PressurePorts[self.closureMuscle_idx].portStatus = PortActions.INFLATE_AND_MODULATE
        self.PressurePorts[self.closureMuscle_idx].commandedPressure = PVal  # value in psi

        self.PressurePorts[self.JawPos[0]].portStatus = PortActions.INFLATE_AND_STOP
        self.PressurePorts[self.JawPos[0]].commandedPressure = self.commandedPosition["Jaw1_psi"]  # value in psi
        self.PressurePorts[self.JawPos[1]].portStatus = PortActions.INFLATE_AND_STOP
        self.PressurePorts[self.JawPos[1]].commandedPressure = self.commandedPosition["Jaw2_psi"]  # value in psi
        self.PressurePorts[self.JawPos[2]].portStatus = PortActions.INFLATE_AND_STOP
        self.PressurePorts[self.JawPos[2]].commandedPressure = self.commandedPosition["Jaw3_psi"]  # value in psi

        byteList = self.ConstructPortCommand()
        numBytes = self.sendCommunicationArray(byteList=byteList)
        self.logger.debug("Bytes sent:%i" % (numBytes))

        # read serial data
        # self.readSerialData()

        ChP = self.getJawChangePressureVals()
        self.changeInPressure = ChP
        self.logger.debug("Change in pressure: " + ','.join([str(x) for x in ChP]))



        #force_vec=[x*5 if x>0.2 else 0 for x in ChP] #threshold for contact

    def ReadGrasperData(self):
        PVal = self.GetPressureFromPosition(self.commandedPosition["ClosureDistance_mm"]) #get the pressure value in psi

        self.logger.debug("CommandedPressure: "+str(PVal))

        self.PressurePorts[self.closureMuscle_idx].portStatus = PortActions.INFLATE_AND_MODULATE
        self.PressurePorts[self.JawPos[0]].portStatus = PortActions.HOLD
        self.PressurePorts[self.JawPos[1]].portStatus = PortActions.HOLD
        self.PressurePorts[self.JawPos[2]].portStatus = PortActions.HOLD



        byteList = self.ConstructPortCommand()
        numBytes = self.sendCommunicationArray(byteList=byteList)
        self.logger.debug("Bytes sent:%i" % (numBytes))

        # read serial data
        tic = time.time()
        self.readSerialData()
        toc = time.time()
        self.logger.debug("Time for Serial Read %f"%(toc-tic))

        ChP = self.getJawChangePressureVals()
        self.changeInPressure = ChP
        self.logger.debug("Change in pressure: "+','.join([str(x) for x in ChP]))

    def sendCommunicationArray(self,startDelim = None,byteList = None, endDelim = None): #send list of bytearrays over serial with start and stop delim
        if startDelim is None:
            startDelim = self.startChar

        if endDelim is None:
            endDelim = self.endChar

        numBytesSent = 0 #keep track of number of bytes sent
        byteArr = bytearray()

        val = startDelim.encode('utf-8')
        #other possibility: byteArr.append(bytes.fromhex("ff"))
        byteArr.extend(val)
        numBytesSent += len(val)

        byteArr.extend(byteList)
        numBytesSent += len(byteList)

        val = endDelim.encode('utf-8')
        byteArr.extend(val)
        numBytesSent += len(val)

        self.ser.write(byteArr)

        #format(int.from_bytes(byteArr,sys.byteorder),"b")

        return(numBytesSent)


    def ConstructPortCommand(self):

        hold_and_actuate_array = bytearray()  # six bytes of  : <aaaabbbb><ccccdddd>...<kkkkllll> where aaaa corresponds to a value between 0 and 15 in the PortActions class above for port 0, bbbb corresponds to port 1, kkkk corresponds to port 10 and llll corresponds to port 11
        numBytes = int(np.ceil(self.numPorts * 4 / 8)) #should be 6

        type_of_action_array = bytearray(int(numBytes))  # get the number of bytes to hold a 4 bit value for each of the ports which will be actuated
        PressureVal = bytearray()  # empty array to fill in with values for the ports later where the value sent is between 0 and 127, and corresponds to the following psi value: max_pressure_psi/floatPressure

        numActuate = 0

        for i, (k, v) in enumerate(self.PressurePorts.items()):


            tempAct = int.from_bytes(type_of_action_array, sys.byteorder) | (v.portStatus.value << (i) * 4)
            type_of_action_array = tempAct.to_bytes(numBytes, byteorder=sys.byteorder)  # windows is little endian , so x[0] is the least significant byte

            if v.portStatus.value == PortActions.INFLATE_AND_MODULATE.value or v.portStatus.value == PortActions.INFLATE_AND_STOP.value:
                numActuate += 1  # increment the number of ports that will be actuated
                PressureVal.extend( round(v.commandedPressure*255/v.maxPressure).to_bytes(1,sys.byteorder) )  # send in the pressure value #only the first byte, i.e. PressureVal[0], contains information


        BytesToSend = bytearray()
        BytesToSend.extend(type_of_action_array)
        BytesToSend.extend(PressureVal)
        return (BytesToSend)


    def readSerialData(self): #inspired by: https://be189.github.io/lessons/14/asynchronous_streaming.html
        # protocol is: > PROTOCOL_BYTE SIZE_BYTE |PAYLOAD of SIZE BYTES| < #https://forum.arduino.cc/t/sending-raw-binary-data/694724
        if self.ser.inWaiting()>0: #if there is data to read
            numBytes = self.ser.inWaiting()
            xd = self.ser.read(numBytes) #read the entire buffer


            totalBuffer = bytearray()
            totalBuffer.extend(self.prevBuffer)
            totalBuffer.extend(xd) #total buffer is the unprocessed data from last round with the new data from this round

            #check to see if the byte(s) representing the start of the communication is present and if the byte(s) representing the end of the communication is present
            rePayload = re.compile(b'.*?>!(?P<Payload>.*?)!<.*?') #replace >> and << with the start and end indicator pos
            payload = rePayload.finditer(totalBuffer)
            index_StartStop = {"start":[],"stop":[]}

            for m in payload: #iterate through matches representing the payload
                payload_m = m.group('Payload')
                index_StartStop["start"].append(m.start('Payload'))
                index_StartStop["stop"].append(m.end('Payload')) #store index of the start and stop of the match.  Will use later to see if there are unprocessed data at the end of the string

                #To DO: sanity check to determine if the number of bytes is correct
                payloadRE = re.compile(b'(?P<ProtocolByte>.)(?P<PayloadSize>.)(?P<Payload>.+)')
                payloadRes = payloadRE.search(payload_m)

                protocolType = None
                numBytes = None
                payload = None
                
                if payloadRes is not None: #if the structure of the payload is in the expected form, then extract the number of bytes etc.
                    protocolType = int.from_bytes(payloadRes.group('ProtocolByte'),sys.byteorder)
                    numBytes = int.from_bytes(payloadRes.group('PayloadSize'),sys.byteorder)
                    payload = payloadRes.group('Payload')

                if payload is not None:
                    if numBytes == len(payload):
                        self.logger.info('Payload matches the expected number of bytes')
                        self.processData(protocolType, numBytes, payload)

                    else:
                        self.logger.error('Warning: Payload size does not match the expected number of bytes\n Protocol Type %s, Numbytes expected %s, Length of payload %s '%(str(protocolType),str(numBytes),str(len(payload))))
                        self.logger.debug(payload)





                #To DO: if stop index is shorter than the length of the total payload, then need to store that for processing in the next time step

                #To DO: If the last stop index is exactly at the end of the total payload, reset self.prevBuffer to be empty.


                
            


            # startpos = totalBuffer.find(self.startChar.encode('utf-8'))
            # endpos = totalBuffer.find(self.endChar.encode('utf-8'))
            #
            # if startpos ~= -1 and endpos ~= -1:
            #     #If both the start and end indicators are present, then process the data.
            #     self.processData(xd)

            #If the start is present but not the end, then add that data to the buffer

    
    def processData(self,protocolType=None, numBytes=None, payload=None):

        if protocolType is not None:
            match protocolType:
                case 0: #pressure values
                    len_payload = math.floor((len(payload)))
                    data=[struct.unpack('f', payload[x:x + 4])[0] for x in range(0, len_payload,4)] #for floats representing pressure values of each port


                    for count, val in enumerate(data):
                        self.PressureArray[count].append(data[count])

                    self.logger.debug("Data for Case 0 (pressure values): " + ','.join([str(x) for x in data]))


                case 1: #strings
                    #for strings
                    self.logger.debug("Payload Case 1 (strings): " + payload.decode())


                case _:
                    #action - default
                    pass

    async def ReadSensorValues(self,number_avg = 1, loop_delay = 0.001): #convenience function to get the pressure values
        jawPressure = np.array([0 for x in self.JawPos])
        closurePressure = 0
        await asyncio.sleep(0.001)


        for i in range(number_avg):
            self.ReadGrasperData()
            jawPressure_r = np.array(self.getJawChangePressureVals())
            self.logger.debug("Read sensor values loop %i, jaw pressures in psi %f,%f,%f"%(i,*jawPressure_r))
            jawPressure = jawPressure_r+jawPressure
            closurePressure = self.PressureArray[self.closureMuscle_idx][-1] + closurePressure
            self.logger.debug("Closure Muscle loop %i, closure muscle pressure in psi %f" % (i, closurePressure))
            await asyncio.sleep(loop_delay)


        jawPressure = jawPressure/number_avg
        closurePressure = closurePressure/number_avg

        return(jawPressure,closurePressure)






























