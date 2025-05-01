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

p.setGravity(0, 0, -10)

planeOrn = [0, 0, 0, 1]  # p.getQuaternionFromEuler([0.3,0,0])
planeId = p.loadURDF("plane.urdf", [0, 0, -1], planeOrn)

Lscale =10
#SoftJaw = p.loadURDF("URDF/SoftGrasperAssembly/urdf/SoftJaw.urdf",[-5, -5, 1],globalScaling=0.05)
SoftSupportInit = p.loadURDF("URDF/SoftGrasperAssembly/urdf/SoftGrasperAssembly.urdf",[0, 0, 0],globalScaling=Lscale,useFixedBase = True,  flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
SoftJawArr = []

for k in range(0,p.getNumJoints(SoftSupportInit)):
    JointI = p.getJointInfo(SoftSupportInit,k)
    if re.match("SJ\d",JointI[1].decode() ) != None: #check if Joint Name is SJ
        posInfo=p.getLinkState(SoftSupportInit,k)
        SoftJaw = p.loadSoftBody("URDF/SoftGrasperAssembly/meshes/SoftJaw1__sf.obj", simFileName = "URDF/SoftGrasperAssembly/meshes/SoftJaw1.vtk", basePosition=posInfo[4],baseOrientation=posInfo[5], scale=Lscale, mass=0.1, useNeoHookean=1,
                                NeoHookeanMu=400, NeoHookeanLambda=600, NeoHookeanDamping=0.1, useSelfCollision=1,
                                frictionCoeff=.5, collisionMargin=0.0001)
        SoftJawArr.append(SoftJaw)

boxId = p.loadURDF("cube.urdf", posInfo[4], useMaximalCoordinates=True,globalScaling=0.06*Lscale)
#sleep(2)
p.setTimeStep(0.001)
p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)


# Read csv with all the deformable points
DF=pd.read_csv('SoftJaw_Sealed_NoCavity_PinNodes.csv',header=9)
IDvals = DF.ID.to_list()
p.removeBody(SoftSupportInit)
SoftSupport = p.loadURDF("URDF/SoftGrasperAssembly/urdf/SoftGrasperAssembly_NoSoftJaw.urdf",[0, 0, 0],globalScaling=Lscale,useFixedBase = True,  flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_MATERIAL_COLORS_FROM_MTL)

for k in range(0,p.getNumJoints(SoftSupportInit)):
    JointI = p.getJointInfo(SoftSupport,k)
    if re.match("LinearJaw\d",JointI[1].decode() ) != None: #check if Joint Name is SJ1,SJ2 etc.
        for ii in IDvals:
            p.createSoftBodyAnchor(SoftJawArr[k], ii, SoftSupport, k, [0, 0, 0])






p.setGravity(0, 0, -10)

Jaw1Pos = p.addUserDebugParameter("Jaw1Pos", -1, 1, 0)
Jaw2Pos = p.addUserDebugParameter("Jaw2Pos", -1, 1, 0)
Jaw3Pos = p.addUserDebugParameter("Jaw3Pos", -1, 1, 0)


while p.isConnected():

    # there can be some artifacts in the visualizer window,
    # due to reading of deformable vertices in the renderer,
    # while the simulators updates the same vertices
    # it can be avoided using
    # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    # but then things go slower

    # sleep(1./240.)
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)


    J1p = p.readUserDebugParameter(Jaw1Pos)
    J2p = p.readUserDebugParameter(Jaw2Pos)
    J3p = p.readUserDebugParameter(Jaw3Pos)
    p.setJointMotorControl2(SoftSupport,
                                            0,
                                            p.POSITION_CONTROL,
                                            targetPosition=J1p,
                                            force=100,
                                            maxVelocity=10)

    p.setJointMotorControl2(SoftSupport,
                                         1,
                                         p.POSITION_CONTROL,
                                         targetPosition=J1p,
                                         force=100,
                                         maxVelocity=10)

    p.setJointMotorControl2(SoftSupport,
                                         2,
                                         p.POSITION_CONTROL,
                                         targetPosition=J1p,
                                         force=100,
                                         maxVelocity=10)

    p.stepSimulation()

# p.resetSimulation()
# p.stopStateLogging(logId)
