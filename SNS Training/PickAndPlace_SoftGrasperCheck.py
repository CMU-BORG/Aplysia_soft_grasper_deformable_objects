import pybullet as p
import pybullet_data
import pathlib
import pandas as pd
import numpy as np
import time


from Gantry.envs.GantrySimulation import GantrySimulation

#########################################################

def pick_and_place():
    gS = GantrySimulation() #gantryURDFfile = "URDF//GrasperAndGantry//urdf//GrasperAndGantry.urdf"
    # add object to the simulation at the center of the plate
    gS.addObjectsToSim("PickupCube", startPos=[0, 0, (0.063+0.02)], mass_kg=1, sizeScaling=0.6,
                       sourceFile=str(pathlib.Path.cwd()/"Gantry\\envs\\URDF\\PickUpObject_URDF\\urdf\\PickUpObject_URDF.urdf"))
    #SoftSupportInit = p.loadURDF("URDF/SoftGrasperAssembly_SimplifiedTilt/urdf/SoftGrasperAssembly_SimplifiedTilt.urdf",
    #                              [0, 0, 0.52816* gS.lengthScale], globalScaling=gS.lengthScale, useFixedBase=False,
    #                              flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT | p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
    #
    #
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath("C://Users//Ravesh//BulletPhysics//bullet3//examples//pybullet//gym//pybullet_data")

    #gS.addObjectsToSim("PickupCube",startPos=[0,0,0.09],mass_kg=1,sizeScaling=0.04,sourceFile="cube.urdf")
    while (not gS.CheckStopSim()):  # check to see if the button was pressed to close the sim

        ts = gS.timeStep  # time step of the simulation in seconds
        nsteps = gS.simCounter  # of simulation steps taken so far
        periodT = 4  # take 4 seconds to complete one sinusoid cycle

        # calculate x position
        jointInfo = gS.bulletClient.getJointInfo(gS.gantryId, gS.GantryLinkIndex_dict[
            "GantryHeadIndex"])  # {"BasePositionIndex":6,"ZAxisBarIndex":1,"GantryHeadId":2,"ClawJointLeftIndex":3,"ClawJointRightIndex":4})
        lowerLimit = jointInfo[
            8]  # getJointInfo does not scale the joint limits properly even when the URDF is imported with a scale factor. While this works out because scaling is applied in StepSim, bear this in mind.
        upperLimit = jointInfo[9]  # same as above
        x = 0.5 * (upperLimit - lowerLimit) * np.sin(2 * np.pi * nsteps * ts / periodT) + 0.5 * (
                    upperLimit - lowerLimit) + lowerLimit

        # calculate y position
        jointInfo = gS.bulletClient.getJointInfo(gS.gantryId, gS.GantryLinkIndex_dict[
            "BasePositionIndex"])  # {"BasePositionIndex":6,"ZAxisBarIndex":1,"GantryHeadId":2,"ClawJointLeftIndex":3,"ClawJointRightIndex":4})
        lowerLimit = jointInfo[
            8]  # getJointInfo does not scale the joint limits properly even when the URDF is imported with a scale factor. While this works out because scaling is applied in StepSim, bear this in mind.
        upperLimit = jointInfo[9]  # same as above
        y = 0.5 * (upperLimit - lowerLimit) * np.sin(2 * np.pi * nsteps * ts / periodT) + 0.5 * (
                upperLimit - lowerLimit) + lowerLimit

        # calculate z position
        jointInfo = gS.bulletClient.getJointInfo(gS.gantryId, gS.GantryLinkIndex_dict["ZAxisBarIndex"])
        lowerLimit = jointInfo[
            8]  # getJointInfo does not scale the joint limits properly even when the URDF is imported with a scale factor. While this works out because scaling is applied in StepSim, bear this in mind.
        upperLimit = jointInfo[9]  # same as above
        z = 0.5 * (upperLimit - lowerLimit) * np.sin(2 * np.pi * nsteps * ts / periodT) + 0.5 * (
                upperLimit - lowerLimit) + lowerLimit + 0.2

        # calculate Jaw Position
        jointInfo = gS.bulletClient.getJointInfo(gS.gantryId, gS.Grasper.LinkDict["LinearJaw1"]) #all the jaws should have the same limits and move the same
        lowerLimit = jointInfo[8]  # getJointInfo does not scale the joint limits properly even when the URDF is imported with a scale factor. While this works out because scaling is applied in StepSim, bear this in mind.
        upperLimit = jointInfo[9]  # same as above
        JawRadialPos =0.5* (upperLimit - lowerLimit) * np.sin(2 * np.pi * nsteps * ts / periodT) + 0.5 * (
                upperLimit - lowerLimit) + lowerLimit

        GrasperArguments = {"frictionCoefficient":0.5,"PressureValue":0.5, #change the pressure value to see change in effective stiffness.
                             "TargetJawPosition":JawRadialPos, "MaxJawForce":20, "MaxVel":0.1,
                             "MaxVertForce":100,
                             "TargetVertPosition":0, "MaxVertVel":0.1}

        ArgumentDict = {"x_gantryHead": x, "y_BasePos": y, "z_AxisBar": z, "x_force": 50, "y_force": 500,
                        "z_force": 500, "GrasperArguments": GrasperArguments}

        # ---------step the simulation----------
        gS.stepSim(usePositionControl=True, GUI_override=False, **ArgumentDict)  # pass argument dict to function







    print("Finished")


if __name__ == "__main__":
    pick_and_place()

