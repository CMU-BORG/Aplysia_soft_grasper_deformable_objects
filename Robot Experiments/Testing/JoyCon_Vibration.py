import time

import numpy as np
import pygame

pygame.init()

#initialise the joystick module
pygame.joystick.init()





joysticks = []

# Add Joysticks
for event in pygame.event.get():
    if event.type == pygame.JOYDEVICEADDED:
        joy = pygame.joystick.Joystick(event.device_index)
        joysticks.append(joy)
    # quit program
    if event.type == pygame.QUIT:
        run = False

# Check the joystick properties
for joystick in joysticks:

    print("Battery Level: " + str(joystick.get_power_level()))
    print("Controller Type: " + str(joystick.get_name()))
    print("Number of axes: " + str(joystick.get_numaxes()))
    joystick.rumble(0,0.1,1000)
    time.sleep(1)
    joystick.rumble(1, 1, 5000)
    time.sleep(2)
    joystick.rumble(0.3, 0.3, 5000)
    time.sleep(2)
    joystick.rumble(0.4, 0.4, 1000)
    time.sleep(0.5)
    joystick.rumble(0.3, 0.3, 1000)
    time.sleep(0.5)
    joystick.rumble(0.4, 0.4, 1000)

prevVal = [0 for x in range(0,11)]

while (True):

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            run = False

    for joystick in joysticks:



        if joystick.get_name() == "Nintendo Switch Joy-Con (R)":

            buttonMapping = {"A":1,"B":3,"X":0,"Y":2,"SL":9,"SR":10,"+":6,"Stick In":7, "Home":5,"R":16,"ZR":18}
            #buttonMapping={"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7,"I":8,"J":9,"K":10,"L":11,"M":12,"N":13,"O":14,"P":15,"Q":16,"R":18}
            buttonValues = {k:None for (k,v) in buttonMapping.items()}
            for k,v in buttonMapping.items():
                buttonValues[k] = joystick.get_button(v)

            # player movement with analogue sticks
            horiz_move = joystick.get_axis(0)
            vert_move = joystick.get_axis(1)
            ax3 = joystick.get_axis(4)
            ax4 = joystick.get_axis(5)



            if np.any(np.array(list(buttonValues.values()))!=np.array(prevVal)):

                print(",".join(buttonValues.keys()))
                print(",".join([str(k) for k in buttonValues.values()]))
                print("Analog Stick: horizontal, vertical")
                print(str(horiz_move)+","+str(vert_move))
                print(str(ax3) + "," + str(ax4))

            time.sleep(0.001)
