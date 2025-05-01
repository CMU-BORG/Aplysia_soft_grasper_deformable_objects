import pybullet as p
import pybullet_data
import pathlib
from pathlib import Path
from copy import deepcopy

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


    SNSc = SNScontroller(ControlMode = ControlType.ORIGINAL)
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

    max_jaw_radial_pos_m = None

    #for modifying the time constant of depositing the object
    print ("Time constant of release is %f"%SNSc.controller._inter_layer_1._params["tau"].data[2])
    SNSc.controller._inter_layer_1._params["tau"].data[2] = 1.5
    print("Time constant of release is %f" % SNSc.controller._inter_layer_1._params["tau"].data[2])

    # orig_time_constant_z = deepcopy(controller._inter_layer_1._params["tau"].data[2])
    # print("Time constant of release is %f" % controller._inter_layer_1._params["tau"].data[2])
    #
    # #For opening closing the gripper
    print("Time constant of grasper opening/closing is %f" % SNSc.controller._inter_layer_1._params["tau"].data[3])
    SNSc.controller._inter_layer_1._params["tau"].data[3] = 0.3
    print("Time constant of grasper opening/closing is %f" % SNSc.controller._inter_layer_1._params["tau"].data[3])

    SNSc.controller.F_MAX = np.array([1000,1000,1000])

    while (not gS.CheckStopSim()):  # check to see if the button was pressed to close the sim


        GUIcontrolTarget = gS.bulletClient.readUserDebugParameter(
            gS.GUIcontrols["GUIcontrolId"])
        if GUIcontrolTarget % 2 == 0 and GUI_control is True:
            GUI_control = False
            gS.simCounter = 0
            object_position_list = [0, 0, -0.315]
            target_position_list = [1.5, 1.5, -0.34]

        if GUI_control is False:
            x = gS.bulletClient.getJointState(
                gS.gantryId, gS.GantryLinkIndex_dict["GantryHeadIndex"])[0]
            y = gS.bulletClient.getJointState(
                gS.gantryId, gS.GantryLinkIndex_dict["BasePositionIndex"])[0]
            z = gS.bulletClient.getJointState(
                gS.gantryId, gS.GantryLinkIndex_dict["ZAxisBarIndex"])[0]
            JawRadialPos = gS.bulletClient.getJointState(
                gS.gantryId, gS.gantryLinkDict["SJ1"])[0]

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

            old_grasperContact = grasperContact

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

        if SNSc.neuronset["lift_after_grasp"]>=20:
            print("lift after grasp")
            print(old_grasperContact)
            if max_jaw_radial_pos_m is None:
                max_jaw_radial_pos_m = JawRadialPos_m
            JawRadialPos_m = min(max_jaw_radial_pos_m,JawRadialPos_m)
            print(JawRadialPos_m)

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
        # if SNSc.neuronset["grasp"]>=60:
        #     controller._inter_layer_1._params["tau"].data[2] = 1000
        # else:
        #     controller._inter_layer_1._params["tau"].data[2] = orig_time_constant_z
        #     pass
        #
        if SNSc.neuronset["grasp"]>=20:
            print("Grasp Contact")
            print(old_grasperContact)
            print(grasperContact)

        else:
            print("Not grasp contact")
            print(grasperContact)

        if SNSc.neuronset["lift_after_grasp"]>=20:
            print(old_grasperContact)

        #     print(grasperContact)
        #     controller._inter_layer_1._params["tau"].data[2] = 0.8
        # else:
        #     controller._inter_layer_1._params["tau"].data[2] = orig_time_constant_z
        #     pass
        if SNSc.neuronset["move_to_release"] >= 10:
            if SNSc.neuronset["release"] <= 0:
                pass

        if SNSc.neuronset["release"] <= 0 and SNSc.neuronset["move_to_release"]<=0:
            pass

        if SNSc.lift_after_grasp_started == True:
            pass

        if SNSc.lift_after_release_done == True:
            pass
        # ---------step the simulation----------
        gS.stepSim(usePositionControl=True, GUI_override=False, **ArgumentDict)  # pass argument dict to function
        print

    return positionset, targetpositionset, forceset, neuronset


#########################################################

if __name__ == "__main__":
    positionset, targetpositionset, forceset, neuronset = pick_and_place()