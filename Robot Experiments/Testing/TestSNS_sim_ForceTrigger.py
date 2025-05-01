import pybullet as p
import pybullet_data
import pathlib
from pathlib import Path

import numpy as np


from GUI.EmbeddedSystems.Support.Structures import Point,GrasperContactForce
from GUI.EmbeddedSystems.Gantry.envs.GantrySimulation import GantrySimulation
from GUI.EmbeddedSystems.SNS.SNScontroller import SNScontroller, ControlType
#########################################################

def pick_and_place():
    gS = GantrySimulation()  # gantryURDFfile = "URDF//GrasperAndGantry//urdf//GrasperAndGantry.urdf"
    # add object to the simulation at the center of the plate
    gS.addObjectsToSim("PickupCube", startPos=[0, 0, (0.063 + 0.02)], mass_kg=0.5, sizeScaling=0.6,
                       sourceFile=str(
                           Path(__file__).parent.parent/"GUI\\EmbeddedSystems\\Gantry\\envs\\URDF\\PickUpObject_URDF\\urdf\\PickUpObject_URDF.urdf"))
    # SoftSupportInit = p.loadURDF("URDF/SoftGrasperAssembly_SimplifiedTilt/urdf/SoftGrasperAssembly_SimplifiedTilt.urdf",
    #                              [0, 0, 0.52816* gS.lengthScale], globalScaling=gS.lengthScale, useFixedBase=False,
    #                              flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT | p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
    #
    #
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath("C://Users//Ravesh//BulletPhysics//bullet3//examples//pybullet//gym//pybullet_data")


    SNSc = SNScontroller(ControlMode = ControlType.NORMAL)
    SNSc.initialize_controller()


    positionset = []
    targetpositionset = []
    forceset = []
    neuronset = []

    GUI_control = True
    logArray = []

    object_position_list = [0,0,0]
    target_position_list = [0.15,0.15,-0.34]
    grasperPosition = Point(0,0,0)
    grasperContact = GrasperContactForce(0,0,0)

    cmd_grasperPos_m = Point(0,0,0)
    JawRadialPos_m = 0

    ts = gS.timeStep
    nsteps = 0



    while (not gS.CheckStopSim()):  # check to see if the button was pressed to close the sim


        GUIcontrolTarget = gS.bulletClient.readUserDebugParameter(
            gS.GUIcontrols["GUIcontrolId"])
        if GUIcontrolTarget % 2 == 0 and GUI_control is True:
            GUI_control = False
            gS.simCounter = 0
            object_position_list = [0, 0, -0.315]
            target_position_list = [1.5, 1.5, -0.34]

        if GUI_control is False: #use SNS
            x = gS.bulletClient.getJointState(
                gS.gantryId, gS.GantryLinkIndex_dict["GantryHeadIndex"])[0]
            y = gS.bulletClient.getJointState(
                gS.gantryId, gS.GantryLinkIndex_dict["BasePositionIndex"])[0]
            z = gS.bulletClient.getJointState(
                gS.gantryId, gS.GantryLinkIndex_dict["ZAxisBarIndex"])[0]
            # JawRadialPos = gS.bulletClient.getJointState(
            #     gS.gantryId, gS.gantryLinkDict["SJ1"])[0]

            grasperPosition = Point(x,y,z)

            force_feedback_1 = gS.bulletClient.getContactPoints(
                gS.gantryId, gS.objects["PickupCube"].objId, gS.gantryLinkDict["SJ1"], -1)
            force_feedback_2 = gS.bulletClient.getContactPoints(
                gS.gantryId, gS.objects["PickupCube"].objId, gS.gantryLinkDict["SJ2"], -1)
            force_feedback_3 = gS.bulletClient.getContactPoints(
                gS.gantryId, gS.objects["PickupCube"].objId, gS.gantryLinkDict["SJ3"], -1)




            commandPosition,JawRadialPos_m = SNSc.SNS_forward(grasperPos_m = grasperPosition, grasperContact = grasperContact,
                                      objectPos_m = Point(*object_position_list),
                                      targetPos_m=Point(*target_position_list))

            cmd_grasperPos_m = Point(*commandPosition)

            if len(force_feedback_1) != 0:
                force_1 = np.linalg.norm(sum(np.array([np.array(x[7]) * x[9] for x in force_feedback_1])), 2)
            else:
                force_1 = 0
            if len(force_feedback_2) != 0:
                force_2 = np.linalg.norm(sum(np.array([np.array(x[7]) * x[9] for x in force_feedback_2])), 2)
            else:
                force_2 = 0
            if len(force_feedback_3) != 0:
                force_3 = np.linalg.norm(sum(np.array([np.array(x[7]) * x[9] for x in force_feedback_3])), 2)
            else:
                force_3 = 0

            grasperContact = GrasperContactForce(force_1, force_2, force_3)

        ts = gS.timeStep  # time step of the simulation in seconds
        nsteps = gS.simCounter  # of simulation steps taken so far


        GrasperArguments = {"frictionCoefficient": 1, "PressureValue": 2.5,
                            # change the pressure value to see change in effective stiffness.
                            "TargetJawPosition": JawRadialPos_m, "MaxJawForce": 20, "MaxVel": 0.1,
                            "MaxVertForce": 100,
                            "TargetVertPosition": 0, "MaxVertVel": 0.1}

        ArgumentDict = {"x_gantryHead": cmd_grasperPos_m.x,
                        "y_BasePos": cmd_grasperPos_m.y,
                        "z_AxisBar": cmd_grasperPos_m.z,
                        "x_force": 50, "y_force": 500,
                        "z_force": 500, "GrasperArguments": GrasperArguments}

        # ---------step the simulation----------
        gS.stepSim(usePositionControl=True, GUI_override=False, **ArgumentDict)  # pass argument dict to function


    return positionset, targetpositionset, forceset, neuronset


#########################################################

if __name__ == "__main__":
    positionset, targetpositionset, forceset, neuronset = pick_and_place()