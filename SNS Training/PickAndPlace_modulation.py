import pybullet as p
import pybullet_data
import pathlib
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Gantry.envs.GantrySimulation import GantrySimulation
from Gantry.controller.SNS_layer import SNS_layer, SENSORY_LAYER_1_INPUT_SIZE, SENSORY_LAYER_1_SIZE, SENSORY_LAYER_2_INPUT_SIZE, SENSORY_LAYER_2_SIZE, THETA_MAX, THETA_MIN, F_MAX, F_MIN, sensory_layer_1, sensory_layer_2, R, perceptor, controller

#########################################################

def pick_and_place():
    gS = GantrySimulation() #gantryURDFfile = "URDF//GrasperAndGantry//urdf//GrasperAndGantry.urdf"
    # add object to the simulation at the center of the plate

    
    gS.addObjectsToSim("PickupCube", startPos=[0, -0.175 * 0, (0.063+0.02)], mass_kg=1, sizeScaling=0.6,
                       sourceFile=str(pathlib.Path.cwd()/"Gantry\\envs\\URDF\\PickUpObject_URDF\\urdf\\PickUpObject_URDF.urdf"))

    #SoftSupportInit = p.loadURDF("URDF/SoftGrasperAssembly_SimplifiedTilt/urdf/SoftGrasperAssembly_SimplifiedTilt.urdf",
    #                              [0, 0, 0.52816* gS.lengthScale], globalScaling=gS.lengthScale, useFixedBase=False,
    #                              flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT | p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
    #
    #
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath("C://Users//Ravesh//BulletPhysics//bullet3//examples//pybullet//gym//pybullet_data")

    positionset = []
    targetpositionset = []
    forceset = []
    neuronset = []

    GUI_control = True

    while (not gS.CheckStopSim()):  # check to see if the button was pressed to close the sim
        timeStart=time.perf_counter()

        GUIcontrolTarget = gS.bulletClient.readUserDebugParameter(
            gS.GUIcontrols["GUIcontrolId"])
        if GUIcontrolTarget % 2 == 0 and GUI_control is True:
            GUI_control = False
            gS.simCounter = 0
            object_position = torch.Tensor([0, 0, -0.34]).unsqueeze(dim=0)
            target_position = torch.Tensor(
                [0.15, 0.15, -0.34]).unsqueeze(dim=0)

        ts = gS.timeStep  # time step of the simulation in seconds
        nsteps = gS.simCounter  # of simulation steps taken so far
        timev = ts*nsteps
        x = gS.bulletClient.getJointState(
            gS.gantryId, gS.GantryLinkIndex_dict["GantryHeadIndex"])[0]
        y = gS.bulletClient.getJointState(
            gS.gantryId, gS.GantryLinkIndex_dict["BasePositionIndex"])[0]
        z = gS.bulletClient.getJointState(
            gS.gantryId, gS.GantryLinkIndex_dict["ZAxisBarIndex"])[0]
        # JawRadialPos = gS.bulletClient.getJointState(
        #     gS.gantryId, gS.gantryLinkDict["SJ1"])[0]
        force_feedback_1 = gS.bulletClient.getContactPoints(
            gS.gantryId, gS.objects["PickupCube"].objId, gS.gantryLinkDict["SJ1"], -1)
        force_feedback_2 = gS.bulletClient.getContactPoints(
            gS.gantryId, gS.objects["PickupCube"].objId, gS.gantryLinkDict["SJ2"], -1)
        force_feedback_3 = gS.bulletClient.getContactPoints(
            gS.gantryId, gS.objects["PickupCube"].objId, gS.gantryLinkDict["SJ3"], -1)
        if len(force_feedback_1) != 0:
            force_1 = np.linalg.norm(sum(np.array([np.array(x[7])*x[9] for x in force_feedback_1])),2)
        else:
            force_1 = 0
        if len(force_feedback_2) != 0:
            force_2 = np.linalg.norm(sum(np.array([np.array(x[7])*x[9] for x in force_feedback_2])),2)
        else:
            force_2 = 0
        if len(force_feedback_3) != 0:
            force_3 = np.linalg.norm(sum(np.array([np.array(x[7])*x[9] for x in force_feedback_3])),2)
        else:
            force_3 = 0
        gripper_position = torch.Tensor([x, y, z]).unsqueeze(dim=0)
        force = torch.Tensor([force_1, force_2, force_3]).unsqueeze(dim=0)
        #print("force_1:",str(force_1))
        #print("force_2:",str(force_2))
        #print("force_3:",str(force_3))

        if GUI_control is False:
            commands = perceptor.forward(
                gripper_position, object_position, target_position, force)
            [move_to_pre_grasp, move_to_grasp, grasp, lift_after_grasp, move_to_pre_release,
                move_to_release, release, lift_after_release] = commands.squeeze(dim=0).numpy()
            [x_d, y_d, z_d, JawRadialPos] = controller.forward(
                object_position, target_position, commands).numpy()
            print("JawPos:",str(JawRadialPos))
            if lift_after_release > 10:
                object_position = torch.Tensor([0, 0, 0]).unsqueeze(dim=0)

            positionset.append([x,y,z])
            targetpositionset.append([x_d,y_d,z_d])
            forceset.append([force_1, force_2, force_3])
            neuronset.append([move_to_pre_grasp, move_to_grasp, grasp, lift_after_grasp, move_to_pre_release, move_to_release, release, lift_after_release])

        else:
            [x_d, y_d, z_d, JawRadialPos] = [0, 0, 0, 0]


        GrasperArguments = {"frictionCoefficient":1,"PressureValue":2.5, #change the pressure value to see change in effective stiffness.
                             "TargetJawPosition":JawRadialPos, "MaxJawForce":20, "MaxVel":0.1,
                             "MaxVertForce":100,
                             "TargetVertPosition":0, "MaxVertVel":0.1}

        ArgumentDict = {"x_gantryHead": x_d, "y_BasePos": y_d, "z_AxisBar": z_d, "x_force": 50, "y_force": 500,
                        "z_force": 500, "GrasperArguments": GrasperArguments}

        # ---------step the simulation----------
        gS.stepSim(usePositionControl=True, GUI_override=False, **ArgumentDict)  # pass argument dict to function

    
    return positionset, targetpositionset, forceset, neuronset

#########################################################

if __name__ == "__main__":
    positionset, targetpositionset, forceset, neuronset = pick_and_place()

    # position_set = np.array(positionset)
    # target_position_set = np.array(targetpositionset)
    # force_set = np.array(forceset)
    # neuron_set = np.array(neuronset)
    # neuron_set[neuron_set<0] = 0

    # plt.figure()
    # plt.subplot(311)
    # plt.plot(position_set[:,-1])
    # plt.plot(target_position_set[:,-1])

    # plt.subplot(312)
    # plt.plot(force_set)

    # plt.subplot(313)
    # plt.plot(np.argmax(neuron_set,axis=1))
    # #plt.plot(neuron_set[:,6])
    # plt.show()