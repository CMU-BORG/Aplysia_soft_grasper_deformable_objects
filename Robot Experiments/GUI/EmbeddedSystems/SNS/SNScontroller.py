import pathlib
import pandas as pd
import numpy as np
import time
import torch



from ..Support.Structures import Point,Velocity,Acceleration,GrasperContactForce
import logging
from pathlib import Path
from datetime import datetime
import sys

from enum import Enum

from .controller.SNS_layer import perceptor as perceptor_original, controller as controller_original #original_controller
from .controller_ForceTrigger.SNS_layer import R, perceptor as perceptor_FT, controller_open_loop, controller_closed_loop_v1, SNS_Control_closed_loop_v1, controller_closed_loop_v2, SNS_Control_closed_loop_v2
from .controller_modulation_open_loop.SNS_layer import Create_Open_Loop_Modulation, SNS_Control_modulation, SNS_Control_closed_loop_modulation
from .controller_modulation.SNS_layer import perceptor_closed_loop_modulation, controller_closed_loop_v2 as controller_closed_loop_v2_modulation, SNS_Control_closed_loop_v2 as SNS_Control_closed_loop_v2_modulation

class ControlType(Enum):
    NORMAL = 0 #close until contact
    OPEN_LOOP = 1 #close until you reach a certain position
    FORCE_INHIBIT = 2 #jaw interneuron has an inhibitory connection from the force feedback to prevent oversqueezing
    FORCE_CAP = 3 #Cap the jaw position to what was triggered after transition from grasp to lift-after-grasp
    MODULATE_OPEN_LOOP = 4 #Modulate the jaw force by increasing the force until it grasps the object
    MODULATE_FORCE_THRESHOLD = 5  # Modulate the jaw force by increasing the threshold until it grasps the object
    ORIGINAL = 6
class SNScontroller:

    def __init__(self,objectPos_m = Point(0,0,0), targetPos_m = Point(0,0,0), ControlMode = ControlType.ORIGINAL,
                 force_threshold_gain=1, inhibitory_gain=1, grasper_closing_speed=1, zero_time_constant=False,
                 modulation_radial_scaling = 0.2, modulation_mod_gain = 5,  modulation_sensory_tau = 0.05,
                 modulation_threshold_gain = 0.1):
        self.neuronset = {
            "move_to_pre_grasp":0,
            "move_to_grasp":0,
            "grasp":0,
            "lift_after_grasp":0,
            "move_to_pre_release":0,
            "move_to_release":0,
            "release":0,
            "lift_after_release":0
        }

        #self.logger = None
        #self.setupLogger()

        self.first_attempt = True

        self.lift_after_release_done = False #set to true when lifted after release neuron exceeds 10.  Necessary to set the object pos to 0,0,0 to trigger the final phases of motion
        self.lift_after_grasp_started = False #set to true when lift after grasp has started for the first time, otherwise set to false
        self.lift_after_grasp_done = False
        self.release_started = False #set to true when the release has started
        self.object_grasped_phase = False #set to true when object is still being grasped, i.e .in grasp, lift_after_grasp, move_to_pre_release or move_to_release are >= 20
        self.motion_complete = False #set to true when motion is complete

        self.object_position = torch.Tensor(list(objectPos_m)).unsqueeze(dim=0)
        self.target_position = torch.Tensor(list(targetPos_m)).unsqueeze(dim=0)

        self.cmd_position_m = Point(0,0,0)
        self.JawRadialPos_m = 0
        self.z_offset = 0

        self.num_grasp_attempts = 0

        #self.ModulateSNS = ModulateSNS #use normal SNS without modulation of contact if False

        self.ControlMode = ControlMode

        self.SNS_release_done = False #for use in the real ganty control. Set to true when trigger the SNS release the 1st time so it doesn't wait 8 seconds for each part of the release phase

        self.controller = None
        self.perceptor = None

        '''Import appropriate SNS controller'''

        self.force_threshold_gain = force_threshold_gain
        self.inhibitory_gain = inhibitory_gain
        self.grasper_closing_speed = grasper_closing_speed
        self.zero_time_constant = zero_time_constant

        self.modulation_radial_scaling = modulation_radial_scaling
        self.modulation_mod_gain = modulation_mod_gain
        self.modulation_sensory_tau = modulation_sensory_tau
        self.modulation_threshold_gain = modulation_threshold_gain






    def initialize_controller(self, reset = True):
        match self.ControlMode:

            case ControlType.MODULATE_OPEN_LOOP:

                self.controller, self.perceptor = Create_Open_Loop_Modulation(self.modulation_radial_scaling)
                #self.controller = controller_modulation
                #self.perceptor = perceptor_modulation

            case ControlType.MODULATE_FORCE_THRESHOLD:
                self.controller = controller_closed_loop_v2_modulation
                self.perceptor = perceptor_closed_loop_modulation

            case ControlType.FORCE_INHIBIT:
                self.controller = controller_closed_loop_v1
                self.perceptor = perceptor_FT

            case ControlType.FORCE_CAP:
                self.controller = controller_closed_loop_v2
                self.perceptor = perceptor_FT

            case ControlType.ORIGINAL:
                self.controller = controller_original
                self.perceptor = perceptor_original
                reset = False

            case ControlType.OPEN_LOOP:
                self.controller = controller_original
                self.perceptor = perceptor_original
                reset = False #don't reset the perceptor or the controller because otherwise does off center move

            case _:  # for normal
                self.controller = controller_open_loop
                self.perceptor = perceptor_FT

        if reset == True:
            self.perceptor.reset()
            self.controller.reset()


    def setupLogger(self):
        ##### Set up logging ####
        logger_soft = logging.getLogger(__name__)

        fname = Path(__file__).parents[3].joinpath('datalogs', str(__name__) + datetime.now().strftime(
            "_%d_%m_%Y_%H_%M_%S") + ".txt")

        fh = logging.FileHandler(fname)  # file handler
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(sys.stdout)  # stream handler
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger_soft.setLevel(logging.DEBUG)
        # add the handlers to the logger_soft
        logger_soft.addHandler(fh)
        logger_soft.addHandler(ch)
        self.logger = logger_soft

    def SNS_forward(self,grasperPos_m:Point, grasperContact:GrasperContactForce,objectPos_m:Point = None,
                    targetPos_m:Point = None, useRealGantry = False, force_threshold_gain=None, inhibitory_gain=None, grasper_closing_speed=None, zero_time_constant=None):
        """
        Class to forward SNS

        Parameters
        ----------
        grasperPos_m    :   Point representing the grasper position in x,y,z space.
                            This is in meters, and is referenced with center of the gantry plate

        grasperContact  :   GrasperContactForce representing the force on jaws 1, 2 and 3.

        objectPos_m    :    Point (optional)
                            Defaults to None. If a point is supplied, updates state variable for object position.

        targetPos_m  :      Point (optional)
                            Defaults to None. If a point is supplied, updates state variable for target position.

        force_threshold_gain: float
            force_threshold_gain scales the force threshold. The force feedback
            divided by this parameter is sent to the network so that the
            force threshold = force_threshold_gain * original force threshold. Default is 1

        inhibitory_gain: float
            inhibitory_gain tunes the weight of the inhibitory synapse from the force sensory neuron
            to the grasper interneuron. This synapse only exists in instances of
            SNS_Control_closed_loop_v1. Default is 1

        grasper_closing_speed: float
            grasper_closing_speed determines how fast the grasper closes or opens if
            an instance of SNS_Control_closed_loop_v2 is selected as the controller model. Default is 1

        zero_time_constant: Boolean
            If true, set the time constants of neurons in the jaw loop to zero so that the jaw can respond more quickly.

        """

        # setup time constants and gains
        if isinstance(self.controller, SNS_Control_closed_loop_v1):
            if inhibitory_gain is not None:
                self.inhibitory_gain = inhibitory_gain
            self.controller._inter_layer_1._params["sensory_erev"].data[-1, -1] = -self.inhibitory_gain * R

        if isinstance(self.controller, SNS_Control_closed_loop_v2) or isinstance(self.controller, SNS_Control_closed_loop_v2_modulation):
            if grasper_closing_speed is not None:
                self.grasper_closing_speed = grasper_closing_speed
            self.controller._inter_layer_2._params["tau"].data[-2:] = self.grasper_closing_speed

            #Setting perceptor layer time constants to 0.01

            # self.perceptor._sensory_layer_1._params["tau"].data = 0.01 * torch.ones_like(
            #     self.perceptor._sensory_layer_1._params["tau"].data)
            # self.perceptor._sensory_layer_2._params["tau"].data = 0.01 * torch.ones_like(
            #     self.perceptor._sensory_layer_2._params["tau"].data)
            # self.perceptor._command_layer._params["tau"].data = 0.01 * torch.ones_like(
            #     self.perceptor._command_layer._params["tau"].data)


        if zero_time_constant is not None:
            self.zero_time_constant = zero_time_constant

        if self.zero_time_constant == True:
            self.controller._inter_layer_1._params["tau"].data[-2:] = 0
            self.controller._motor_layer._params["tau"].data[3] = 0


        if force_threshold_gain is not None:
            self.force_threshold_gain = force_threshold_gain




        if objectPos_m is not None and self.lift_after_release_done == False:
            self.object_position = torch.Tensor(list(objectPos_m)).unsqueeze(dim=0) #update the object pos


        if self.neuronset["lift_after_release"]>10:
            self.lift_after_release_done = True

        if self.lift_after_release_done == True:
            self.object_position = torch.Tensor([0, 0, 0]).unsqueeze(dim=0)


        if targetPos_m is not None:
            self.target_position = torch.Tensor(list(targetPos_m)).unsqueeze(dim=0) #update the target pos

        #Before object grasped phase, the target position x and y need to be same as the object position. If in object grasped phase, use the correct x and y
        # if (self.neuronset["move_to_pre_grasp"] >0):
        #     targ_pos = torch.Tensor([self.object_position.squeeze().numpy()[0],
        #                              self.object_position.squeeze().numpy()[1],
        #                              self.target_position.squeeze().numpy()[2]]).unsqueeze(dim=0)
        # else:
        #     targ_pos = self.target_position

        targ_pos = self.target_position

        force = torch.Tensor(list(grasperContact)).unsqueeze(dim=0)*self.force_threshold_gain


        grasperPos_m = Point(grasperPos_m.x,grasperPos_m.y,grasperPos_m.z-self.z_offset)
        gripper_position = torch.Tensor(list(grasperPos_m)).unsqueeze(dim=0)


        if self.ControlMode == ControlType.ORIGINAL or self.ControlMode == ControlType.OPEN_LOOP: #for the original version and the modulation
            commands = self.perceptor.forward(
                gripper_position, self.object_position, targ_pos, force)


            [move_to_pre_grasp, move_to_grasp, grasp, lift_after_grasp, move_to_pre_release,
             move_to_release, release, lift_after_release] = commands.squeeze(dim=0).numpy()

            [x_d, y_d, z_d, JawRadialPos_m] = self.controller.forward(
                self.object_position, targ_pos, commands).numpy()

        elif self.ControlMode == ControlType.FORCE_INHIBIT or self.ControlMode == ControlType.FORCE_CAP \
                or self.ControlMode == ControlType.NORMAL or self.ControlMode == ControlType.MODULATE_OPEN_LOOP or \
                self.ControlMode == ControlType.MODULATE_FORCE_THRESHOLD:
            commands, force_output = self.perceptor.forward(gripper_position, self.object_position,
                                                            targ_pos, force)

            [move_to_pre_grasp, move_to_grasp, grasp, lift_after_grasp, move_to_pre_release,
             move_to_release, release, lift_after_release] = commands.squeeze(dim=0).numpy()

            force_sum = force_output.squeeze(dim=0).numpy()
            if isinstance(self.controller, SNS_Control_closed_loop_v1):
                motor_states = self.controller.forward(self.object_position, targ_pos, commands, force_output)
            else:
                motor_states = self.controller.forward(self.object_position, targ_pos, commands)

            [x_d, y_d, z_d, JawRadialPos_m] = motor_states.numpy()




        cmd_grasperPos_m = Point(x_d, y_d,
                                 z_d + self.z_offset)  # note that this is in meters, and doesn't account for the offset that is considered 0,0,0 on the gantry.  Need to correct for that afterwards


        ### Setup the indicators for the phase of motion###
        if self.neuronset["grasp"] < 20 and grasp >= 20:
            self.num_grasp_attempts = self.num_grasp_attempts + 1
            #print(self.num_grasp_attempts)

        if self.neuronset["grasp"]>=20 and lift_after_grasp >=20 and self.neuronset['lift_after_grasp'] <=10:
            self.lift_after_grasp_started = True #should only trigger once
        else:
            self.lift_after_grasp_started = False

        if self.neuronset["move_to_pre_release"] >= 20:
            self.lift_after_grasp_done = True

        self.object_grasped_phase = (self.neuronset['grasp'] >= 20 or self.neuronset['lift_after_grasp'] >= 20
                        or self.neuronset['move_to_pre_release'] >= 20 or
                self.neuronset['move_to_release'] >= 20) ##these should all only trigger once. Only move to pre grasp, move to grasp, grasp are triggered twice, once in the beginning and once when it returns home.

        if self.neuronset["release"]>=10 and release <=0:
            self.release_started = True

        self.motion_complete = self.lift_after_release_done == True  and self.neuronset["grasp"]>=20 #finished SNS motion

        #if using the real gantry, stop after one grasp attempt
        if useRealGantry == True:

            #print("Grasp: %i , previous: %i"%(self.neuronset["grasp"],grasp))

            if lift_after_release>=20 or move_to_pre_release>=20:
                self.z_offset = 0.07 #add offset to the z so it lifts the object higher

            if move_to_release>=20 or release >= 20:
                self.z_offset = -0.07

            if lift_after_release>=20:
                self.z_offset = 0

            if self.num_grasp_attempts<=1:

                if self.neuronset["lift_after_release"]>=20:
                    JawRadialPos_m = 0
                    cmd_grasperPos_m = self.cmd_position_m #use the last commanded position, don't return to home
                    #self.logger.debug('Lift after release triggered, stay at this position')



            else: #if more than 1 grasp attempted
                JawRadialPos_m = 0
                cmd_grasperPos_m = self.cmd_position_m #use the last commanded position, don't return to home
                #self.logger.debug('More than 1 grasp attempted')




        self.cmd_position_m = cmd_grasperPos_m
        self.JawRadialPos_m = JawRadialPos_m



        #neuron state update
        self.neuronset['move_to_pre_grasp'] = move_to_pre_grasp
        self.neuronset['move_to_grasp'] = move_to_grasp
        self.neuronset['grasp'] = grasp
        self.neuronset['lift_after_grasp'] = lift_after_grasp
        self.neuronset['move_to_pre_release'] = move_to_pre_release
        self.neuronset['move_to_release'] = move_to_release
        self.neuronset['release'] = release
        self.neuronset['lift_after_release'] = lift_after_release





        return(cmd_grasperPos_m,JawRadialPos_m)


