#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright 2017 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

# Author: Ryu Woon Jung (Leon)

#
# *********     Read and Write Example      *********
#
#
# Available Dynamixel model on this example : All models using Protocol 2.0
# This example is designed for using a Dynamixel PRO 54-200, and an USB2DYNAMIXEL.
# To use another Dynamixel model, such as X series, see their details in E-Manual(emanual.robotis.com) and edit below variables yourself.
# Be sure that Dynamixel PRO properties are already set as %% ID : 1 / Baudnum : 1 (Baudrate : 57600)
#

import os

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

from dynamixel_sdk import *                    # Uses Dynamixel SDK library

# Control table address
ADDR_PRO_TORQUE_ENABLE      = 64               # Control table address is different in Dynamixel model
ADDR_PRO_GOAL_POSITION      = 116
ADDR_PRO_PRESENT_POSITION   = 132

# Protocol version
PROTOCOL_VERSION            = 2.0               # See which protocol version is used in the Dynamixel

# Default setting
DXL_ID1                      = 1                 # Dynamixel ID : 1
DXL_ID2                      = 2                 # Dynamixel ID : 1
BAUDRATE                    = 57600             # Dynamixel default baudrate : 57600
DEVICENAME                  = 'COM4'    # Check which port is being used on your controller
                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
DXL_MINIMUM_POSITION_VALUE  = 2000          # Dynamixel will rotate between this value
DXL_MAXIMUM_POSITION_VALUE  = 1500        # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
DXL_MOVING_STATUS_THRESHOLD = 100               # Dynamixel moving status threshold

OPERATING_MODE = 11
CONTROL_MODE = 3

POSITION_PROP_BYTE = 84
POSITION_PROP_GAIN = 100

VELOCITY_PROP_BYTE = 78
VELOCITY_PROP_GAIN = 0

VELOCITY_I_BYTE = 76
VELOCITY_I_GAIN = 0



index = 0
dxl_goal_position = [DXL_MAXIMUM_POSITION_VALUE,DXL_MINIMUM_POSITION_VALUE]         # Goal position
dxl_goal_position_1 = [1500,2000]
dxl_goal_position_2 = [2120,1620]

# Initialize PortHandler instance
# Set the port path
# Get methods and members of PortHandlerLinux or PortHandlerWindows
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
# Set the protocol version
# Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()


# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()

################### For Motor 1
#set operating mode
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID1, OPERATING_MODE, CONTROL_MODE)

# Enable Dynamixel Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID1, ADDR_PRO_TORQUE_ENABLE, TORQUE_ENABLE)



#set gain
dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID1, POSITION_PROP_BYTE, POSITION_PROP_GAIN)


dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID1, VELOCITY_PROP_BYTE, VELOCITY_PROP_GAIN)
dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID1, VELOCITY_I_BYTE, VELOCITY_I_GAIN)

################## For Motor 2
#set operating mode
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID2, OPERATING_MODE, CONTROL_MODE)

# Enable Dynamixel Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID2, ADDR_PRO_TORQUE_ENABLE, TORQUE_ENABLE)



#set gain
dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID2, POSITION_PROP_BYTE, POSITION_PROP_GAIN)


dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID2, VELOCITY_PROP_BYTE, VELOCITY_PROP_GAIN)
dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID2, VELOCITY_I_BYTE, VELOCITY_I_GAIN)


dxl_present_position1, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID1,
                                                                                       ADDR_PRO_PRESENT_POSITION)

dxl_present_position2, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID2,
                                                                                       ADDR_PRO_PRESENT_POSITION)

print("Initial Positions: %d, %d"%(dxl_present_position1,dxl_present_position2))



if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel has been successfully connected")

while 1:
    print("Press any key to continue! (or press ESC to quit!)")
    if getch() == chr(0x1b):
        break

    # Write goal position for Motor 1
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID1, ADDR_PRO_GOAL_POSITION, dxl_goal_position_1[index])
    if dxl_comm_result != COMM_SUCCESS:
        print("%s\n" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s\n" % packetHandler.getRxPacketError(dxl_error))

    # Write goal position for Motor 2
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID2, ADDR_PRO_GOAL_POSITION,
                                                              dxl_goal_position_2[index])
    if dxl_comm_result != COMM_SUCCESS:
        print("%s\n" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s\n" % packetHandler.getRxPacketError(dxl_error))

    Motor1_Fin = False
    Motor2_Fin = False

    while 1:

        # Read present position of Motor 1
        dxl_present_position1, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID1, ADDR_PRO_PRESENT_POSITION)

        if abs(dxl_goal_position_1[index] - dxl_present_position1) > DXL_MOVING_STATUS_THRESHOLD:
            if dxl_comm_result != COMM_SUCCESS:
                print("%s\n" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s\n" % packetHandler.getRxPacketError(dxl_error))

            print("[ID:%03d] GoalPos:%03d  PresPos:%03d\n" % (DXL_ID1, dxl_goal_position_1[index], dxl_present_position1))

            dxl_movingStatus, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(portHandler, DXL_ID1, 123)
            print("Moving Status: %s" % format(dxl_movingStatus, '08b'))

        else:
            Motor1_Fin = True

        # Read present position of Motor 2
        dxl_present_position2, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID2,
                                                                                       ADDR_PRO_PRESENT_POSITION)

        if abs(dxl_goal_position_2[index] - dxl_present_position2) > DXL_MOVING_STATUS_THRESHOLD:
            if dxl_comm_result != COMM_SUCCESS:
                print("%s\n" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s\n" % packetHandler.getRxPacketError(dxl_error))

            print(
                "[ID:%03d] GoalPos:%03d  PresPos:%03d\n" % (DXL_ID2, dxl_goal_position_2[index], dxl_present_position2))

            dxl_movingStatus, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(portHandler, DXL_ID2, 123)
            print("Moving Status: %s" % format(dxl_movingStatus, '08b'))

        else:
            Motor2_Fin = True

        if Motor1_Fin == True and Motor2_Fin == True:
            break

    # Change goal position
    if index == 0:
        index = 1
    else:
        index = 0


# Disable Dynamixel Torque for Motor 1
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID1, ADDR_PRO_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s\n" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s\n" % packetHandler.getRxPacketError(dxl_error))

# Disable Dynamixel Torque For Motor 2
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID2, ADDR_PRO_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s\n" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s\n" % packetHandler.getRxPacketError(dxl_error))

# Close port
portHandler.closePort()
