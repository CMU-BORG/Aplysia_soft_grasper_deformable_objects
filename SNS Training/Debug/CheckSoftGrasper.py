#meshing: https://github.com/bulletphysics/bullet3/issues/2726

import pybullet as p
from time import sleep
import pybullet_data
import pandas as pd

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setAdditionalSearchPath("C://Users//Ravesh//BulletPhysics//bullet3//examples//pybullet//gym//pybullet_data")

p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)

p.setGravity(0, 0, -10)

planeOrn = [0, 0, 0, 1]  # p.getQuaternionFromEuler([0.3,0,0])
planeId = p.loadURDF("plane.urdf", [0, 0, -1], planeOrn)

boxId = p.loadURDF("cube.urdf", [-5, -5, 3], useMaximalCoordinates=True)
#SoftJaw = p.loadURDF("URDF/SoftGrasperAssembly/urdf/SoftJaw.urdf",[-5, -5, 1],globalScaling=0.05)
SoftSupport = p.loadURDF("URDF/SoftGrasperAssembly/urdf/SoftJawSupport.urdf",[-5, -5, -1],globalScaling=0.01)
SoftJaw = p.loadSoftBody("SoftJaw_Sealed_NoCavity.STL__sf.obj", simFileName = "SoftJaw_Sealed_NoCavity.vtk", basePosition=[-5, -5, 0], scale=0.01, mass=4, useNeoHookean=1,
                        NeoHookeanMu=40000, NeoHookeanLambda=600, NeoHookeanDamping=0.1, useSelfCollision=1,
                        frictionCoeff=.5, collisionMargin=0.001)

sleep(2)
p.setTimeStep(0.001)
p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)

# logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "perf.json")

# Read csv with all the deformable points
DF=pd.read_csv('SoftJaw_Sealed_NoCavity_PinNodes.csv',header=9)
IDvals = DF.ID.to_list()
# for ii in IDvals:
#     p.createSoftBodyAnchor(ballId,ii,-1,-1)



p.setGravity(0, 0, -10)
while p.isConnected():
    p.stepSimulation()
    # there can be some artifacts in the visualizer window,
    # due to reading of deformable vertices in the renderer,
    # while the simulators updates the same vertices
    # it can be avoided using
    # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    # but then things go slower

    # sleep(1./240.)
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)




# p.resetSimulation()
# p.stopStateLogging(logId)
