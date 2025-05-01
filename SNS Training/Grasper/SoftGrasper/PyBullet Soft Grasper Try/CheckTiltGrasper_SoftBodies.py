#meshing: https://github.com/bulletphysics/bullet3/issues/2726

import pybullet as p
from time import sleep
import pybullet_data
import pandas as pd
import re

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setAdditionalSearchPath("C://Users//Ravesh//BulletPhysics//bullet3//examples//pybullet//gym//pybullet_data")

p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD) #use FEM



planeOrn = [0, 0, 0, 1]  # p.getQuaternionFromEuler([0.3,0,0])
planeId = p.loadURDF("plane.urdf", [0, 0, 0], planeOrn)

Lscale =10
#SoftJaw = p.loadURDF("URDF/SoftGrasperAssembly/urdf/SoftJaw.urdf",[-5, -5, 1],globalScaling=0.05)

p.setGravity(0, 0, -10*Lscale)

SoftJawArr = []


SoftSupportInit = p.loadURDF("URDF/SoftGrasperAssembly_SimplifiedTilt/urdf/SoftGrasperAssembly_SimplifiedTilt.urdf",[0, 0, 2],globalScaling=Lscale,useFixedBase = True,  flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
SoftJawArr = []
SoftJawDict={}
for k in range(0,p.getNumJoints(SoftSupportInit)):
    JointI = p.getJointInfo(SoftSupportInit,k)
    if re.match("SJ\d",JointI[1].decode() ) is not None: #check if Joint Name is SJ
        posInfo=p.getLinkState(SoftSupportInit,k)
        SoftJaw = p.loadSoftBody("URDF/SoftGrasperAssembly_SimplifiedTilt/meshes/SoftJaw1.vtk", simFileName = "URDF/SoftGrasperAssembly_SimplifiedTilt/meshes/SoftJaw1.vtk", basePosition=posInfo[4],baseOrientation=posInfo[5], scale=Lscale, mass=0.05, useNeoHookean=1,
                                NeoHookeanMu=112686/Lscale, NeoHookeanLambda=100000, NeoHookeanDamping=0.5, useSelfCollision=1,
                                frictionCoeff=1, collisionMargin=0.000001)

        # SoftJaw = p.loadSoftBody("URDF/SoftGrasperAssembly_SimplifiedTilt/meshes/SoftJaw1.vtk",
        #                          simFileName="URDF/SoftGrasperAssembly_SimplifiedTilt/meshes/SoftJaw1.vtk",
        #                          basePosition=posInfo[4], baseOrientation=posInfo[5], scale=Lscale, mass=0.05,springElasticStiffness = 100,springDampingStiffness = 0.1,
        #                          useSelfCollision=1, useNeoHookean = 0,useMassSpring=F,
        #                          frictionCoeff=.5, collisionMargin=0.000001)
        SoftJawArr.append(SoftJaw)
        DynInfo = list(p.getDynamicsInfo(SoftSupportInit, k))

        if DynInfo[8]==-1: #
            DynInfo[8] = 0.1
        if DynInfo[9]==-1:
            DynInfo[9] = 1000

        SoftJawDict[JointI[1].decode()] = {"LinkIndex":k,"Mass":DynInfo[0],"Lateral Friction":DynInfo[1],"Rolling Friction":DynInfo[6],"Spinning Friction":DynInfo[7],"Contact Damping":DynInfo[8],"Contact Stiffness":DynInfo[9]}

boxId = p.loadURDF("cube.urdf", [0,0,0.05*Lscale], useMaximalCoordinates=True,globalScaling=0.04*Lscale)





#sleep(2)
p.setTimeStep(0.0001)
p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
p.changeDynamics(boxId,-1,1)

DF=pd.read_csv('SoftJaw_Sealed_NoCavity_PinNodes.csv',header=9)
IDvals = DF.ID.to_list()
p.removeBody(SoftSupportInit)
SoftSupport = p.loadURDF("URDF/SoftGrasperAssembly_SimplifiedTilt/urdf/SoftGrasperAssembly_SimplifiedTilt_SoftJaw.urdf",[0, 0, 2],globalScaling=Lscale,useFixedBase = True,  flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT | p.URDF_USE_MATERIAL_COLORS_FROM_MTL)

SJiter = iter(SoftJawArr)
for k in range(0,p.getNumJoints(SoftSupport)):
    JointI = p.getJointInfo(SoftSupport,k)
    if re.match("LinearJaw\d",JointI[1].decode() ) != None: #check if Joint Name is SJ1,SJ2 etc.
        SJind = next(SJiter)
        for ii in IDvals:
            p.createSoftBodyAnchor(SJind, ii, SoftSupport, k, [0, 0, 0])





Jaw1Pos = p.addUserDebugParameter("Jaw1Pos", -2, 2, 0)
Jaw2Pos = p.addUserDebugParameter("Jaw2Pos", -1, 1, 0)
Jaw3Pos = p.addUserDebugParameter("Jaw2Pos", -1, 1, 0)
Jaw4Pos = p.addUserDebugParameter("Jaw2Pos", -1, 1, 0)
ContactFrictionSlider = p.addUserDebugParameter("Contact Friction Scale Factor (LogScale)", -2, 2, 0)
ContactStiffnessSlider = p.addUserDebugParameter("Contact Stiffness Scale Factor(LogScale)", -2, 2, 0)
i=0
while p.isConnected():

    # there can be some artifacts in the visualizer window,
    # due to reading of deformable vertices in the renderer,
    # while the simulators updates the same vertices
    # it can be avoided using
    # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    # but then things go slower

    # sleep(1./240.)
    #p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
    J1p = p.readUserDebugParameter(Jaw1Pos)
    J2p = p.readUserDebugParameter(Jaw2Pos)
    J3p = p.readUserDebugParameter(Jaw3Pos)
    J4p = p.readUserDebugParameter(Jaw4Pos)
    FricScale = 10**(p.readUserDebugParameter(ContactFrictionSlider)) #convert from log scale
    StiffScale = 10**(p.readUserDebugParameter(ContactStiffnessSlider)) #convert from log scale

    p.setJointMotorControl2(SoftSupport,
                            0,
                            p.POSITION_CONTROL,
                            targetPosition=J1p,
                            force=1000*Lscale,
                            maxVelocity=10)
    p.setJointMotorControl2(SoftSupport,
                            1,
                            p.POSITION_CONTROL,
                            targetPosition=J2p,
                            force=100*Lscale,
                            maxVelocity=10)

    p.setJointMotorControl2(SoftSupport,
                            2,
                            p.POSITION_CONTROL,
                            targetPosition=J2p,
                            force=100*Lscale,
                            maxVelocity=10)

    p.setJointMotorControl2(SoftSupport,
                            3,
                            p.POSITION_CONTROL,
                            targetPosition=J2p,
                            force=100*Lscale,
                            maxVelocity=10)

    p.stepSimulation()
    i=i+1
    print(i)

# p.resetSimulation()
# p.stopStateLogging(logId)
