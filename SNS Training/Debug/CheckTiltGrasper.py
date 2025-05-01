#meshing: https://github.com/bulletphysics/bullet3/issues/2726

import pybullet as p
from time import sleep
import pybullet_data
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as pltv


from scipy.optimize import fsolve

class ContactForce:
    def __init__(self,coefficients = [0.0416, 0.1791, 0.8912, 5.4641, 1.0647, 0.5050]):
        self.coefficients = coefficients
        self.ZeroDistance = 0

    '''
    Contact Force:
    pressure: value in gauge psi
    displacement: value in mm.  convention is 0 is the surface of the soft jaw, +ve values indicate compression into the jaw, -ve values indicate distances above the jaw
    '''
    def ContactForceFunc(self,pressure=0,displacement=0,coefficients=None):
        b = self.coefficients if coefficients is None else coefficients
        x = displacement
        P = pressure
        F = ((b[5] * P ** b[4] + b[0]) * ((b[1] ** (2 * P)) * x ** 2 + (b[2] ** P) * x + b[3]))
        return F

    def ContactForce_ZeroPos(self,pressure=0, coefficients=None):
        b = self.coefficients if coefficients is None else coefficients
        F_P = lambda x: self.ContactForceFunc(pressure=pressure,displacement=x,coefficients=b)
        zeroloc = fsolve(F_P, 0)
        self.ZeroDistance = zeroloc
        return(zeroloc)
    def ContactForceCalc(self,pressure,displacement,coefficients = None,zeroPos=None):
        b = self.coefficients if coefficients is None else coefficients
        F_val = self.ContactForceFunc(pressure=pressure, displacement=displacement, coefficients=b)
        zPos = self.ContactForce_ZeroPos(pressure=pressure,coefficients = b)if zeroPos is None else zeroPos
        F_val = F_val if (displacement>zPos or zPos>0) else 0
        return F_val

    def ContactStiffnessCalc(self,pressure,displacement,coefficients=None): #Returns in N/mm
        b = self.coefficients if coefficients is None else coefficients
        StiffVal = (b[0] + (pressure**b[4]) *b[5])*(b[2]**pressure + 2*(b[1]**(2*pressure)) * displacement)
        return StiffVal





physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setAdditionalSearchPath("C://Users//Ravesh//BulletPhysics//bullet3//examples//pybullet//gym//pybullet_data")

#p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD) #use FEM




planeOrn = [0, 0, 0, 1]  # p.getQuaternionFromEuler([0.3,0,0])
planeId = p.loadURDF("plane.urdf", [0, 0, 0], planeOrn)

Lscale =10
#SoftJaw = p.loadURDF("URDF/SoftGrasperAssembly/urdf/SoftJaw.urdf",[-5, -5, 1],globalScaling=0.05)

p.setGravity(0, 0, -10*Lscale)

SoftJawArr = []


SoftSupportInit = p.loadURDF("URDF/SoftGrasperAssembly_SimplifiedTilt/urdf/SoftGrasperAssembly_SimplifiedTilt.urdf",[0, 0, 0.2*Lscale],globalScaling=Lscale,useFixedBase = True,  flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT | p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
SoftJawArr = []
SoftJawDict={}
for k in range(0,p.getNumJoints(SoftSupportInit)):
    JointI = p.getJointInfo(SoftSupportInit,k)
    if re.match("SJ\d",JointI[1].decode() ) is not None: #check if Joint Name is SJ
        posInfo=p.getLinkState(SoftSupportInit,k)
        # SoftJaw = p.loadSoftBody("URDF/SoftGrasperAssembly_SimplifiedTilt/meshes/SoftJaw1__sf.obj", simFileName = "URDF/SoftGrasperAssembly_SimplifiedTilt/meshes/SoftJaw1.vtk", basePosition=posInfo[4],baseOrientation=posInfo[5], scale=Lscale, mass=0.1, useNeoHookean=1,
        #                         NeoHookeanMu=400, NeoHookeanLambda=6
        #
        #                         00, NeoHookeanDamping=0.1, useSelfCollision=1,
        #                         frictionCoeff=.5, collisionMargin=0.0001)
        # SoftJawArr.append(SoftJaw)
        DynInfo = list(p.getDynamicsInfo(SoftSupportInit, k))

        if DynInfo[8]==-1: #
            DynInfo[8] = 1
        if DynInfo[9]==-1:
            DynInfo[9] = 0.1

        SoftJawDict[JointI[1].decode()] = {"LinkIndex":k,"Mass":DynInfo[0],"Lateral Friction":DynInfo[1],"Rolling Friction":DynInfo[6],"Spinning Friction":DynInfo[7],"Contact Damping":DynInfo[8],"Contact Stiffness":DynInfo[9],"JointForceTorque":[],
                                           "ContactNumber":[],"IndexContact":[],"NormalForce":[],"FrictionForce1":[],"FrictionForce2":[],
                                           "SpringForce":[],"PenetrationDistance":[],"AppliedForceStart":{0:[],1:[],2:[],3:[]}}
    p.enableJointForceTorqueSensor(SoftSupportInit,k,enableSensor=True)

boxId = p.loadURDF("cube.urdf", [0,0,0], useMaximalCoordinates=True,globalScaling=0.04*Lscale)





#sleep(2)
p.setTimeStep(0.001)
p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)






#p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

Jaw1Pos = p.addUserDebugParameter("Jaw1Pos", -1, 1, 0)
Jaw2Pos = p.addUserDebugParameter("Jaw2Pos", -1, 1, 0)
Jaw3Pos = p.addUserDebugParameter("Jaw2Pos", -1, 1, 0)
Jaw4Pos = p.addUserDebugParameter("Jaw2Pos", -1, 1, 0)
ContactFrictionSlider = p.addUserDebugParameter("Contact Friction Scale Factor (LogScale)", -2, 2, 0)
ContactStiffnessSlider = p.addUserDebugParameter("Contact Stiffness Scale Factor(LogScale)", -2, 2, 0)

PressureSlider = p.addUserDebugParameter("Pressure", 0, 2.5, 0)

ApplyExternalForce = p.addUserDebugParameter("ApplyExternalForce",0,-1,0)

exitButton = p.addUserDebugParameter("Exit",0,-1,0)
initValueButton = p.readUserDebugParameter(exitButton)

ApplyEF =p.readUserDebugParameter(ApplyExternalForce)
MassOfBox=p.getDynamicsInfo(boxId,-1)[0]



# ForceOnBox = p.addUserDebugParameter("ForceOnBox", -200, 200, 0)

index = 0
switchOn = 0

CF = ContactForce() #initialize ContactForce Class
numTsteps = 600

while (p.isConnected()):
    valueButton =p.readUserDebugParameter(exitButton)
    if (valueButton>initValueButton):
        break


    # there can be some artifacts in the visualizer window,
    # due to reading of deformable vertices in the renderer,
    # while the simulators updates the same vertices
    # it can be avoided using
    # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    # but then things go slower

    # sleep(1./240.)
    J1p = p.readUserDebugParameter(Jaw1Pos)
    J2p = p.readUserDebugParameter(Jaw2Pos)
    J3p = p.readUserDebugParameter(Jaw3Pos)
    J4p = p.readUserDebugParameter(Jaw4Pos)
    # appliedBoxForce = p.readUserDebugParameter(ForceOnBox)
    FricScale = 10**(p.readUserDebugParameter(ContactFrictionSlider)) #convert from log scale
    StiffScale = 10**(p.readUserDebugParameter(ContactStiffnessSlider)) #convert from log scale
    NewApplyEF = p.readUserDebugParameter(ApplyExternalForce)

    PressureValue = p.readUserDebugParameter(PressureSlider)


    if NewApplyEF != ApplyEF and switchOn == 0:
        switchOn = index

    p.setJointMotorControl2(SoftSupportInit,
                            0,
                            p.POSITION_CONTROL,
                            targetPosition=J1p,
                            force=20*Lscale,
                            maxVelocity=10)
    p.setJointMotorControl2(SoftSupportInit,
                            1,
                            p.POSITION_CONTROL,
                            targetPosition=J2p,
                            force=20*Lscale,
                            maxVelocity=10)

    p.setJointMotorControl2(SoftSupportInit,
                            3,
                            p.POSITION_CONTROL,
                            targetPosition=J2p,
                            force=20*Lscale,
                            maxVelocity=10)

    p.setJointMotorControl2(SoftSupportInit,
                            5,
                            p.POSITION_CONTROL,
                            targetPosition=J2p,
                            force=20*Lscale,
                            maxVelocity=10)

    #p.applyExternalForce(boxId, -1, [0,0,0], appliedBoxForce, p.LINK_FRAME)
    # print(appliedBoxForce)


    netFricForce = 0
    netTotalForce = 0
    for k,v in SoftJawDict.items():
        DynInfo = p.getDynamicsInfo(SoftSupportInit,v["LinkIndex"])
        #print("%s properties: Mass %f, Lateral Friction %f, Rolling Friction %f, Spinning Friction %f, Contact Damping %f Contact Stiffness %f"%(k,DynInfo[0],DynInfo[1],DynInfo[6],DynInfo[7],DynInfo[8],DynInfo[9]))

        #Change stiffness and friction
        p.changeDynamics(SoftSupportInit,v["LinkIndex"],contactStiffness=v["Contact Stiffness"]*StiffScale,contactDamping = v["Contact Damping"],lateralFriction=v["Lateral Friction"]*FricScale)

        contactPoints = p.getContactPoints(SoftSupportInit, boxId, SoftJawDict[k]["LinkIndex"],-1)
        JointRF = p.getJointState(SoftSupportInit,v["LinkIndex"])
        SoftJawDict[k]["JointForceTorque"].append(list(JointRF[2]))


        for gg in range(0,len(contactPoints)):

            cP = contactPoints[gg]
            #print("%s,pos On SJ: %s, pos on Box: %s, Contact Normal on Box towards SJ: %s, Contact Distance: %f, Normal Force: %f"%(k,(cP[5],),(cP[6],),(cP[7],),cP[8],cP[9]))
            contactF = CF.ContactForceCalc(PressureValue, -cP[8]*1000/Lscale,zeroPos=zP)

            ## Assign the lateral friction forces to the vector

            SoftJawDict[k]["ContactNumber"].append(gg)
            SoftJawDict[k]["IndexContact"].append(index)
            SoftJawDict[k]["NormalForce"].append(cP[9]*np.array(cP[7]))
            SoftJawDict[k]["FrictionForce1"].append(cP[10] * np.array(cP[11]))
            SoftJawDict[k]["FrictionForce2"].append(cP[12] * np.array(cP[13]))
            SoftJawDict[k]["SpringForce"].append(cP[8]*v["Contact Stiffness"]*StiffScale)
            SoftJawDict[k]["PenetrationDistance"].append(cP[8])
            netFricForce += cP[10] * np.array(cP[11]) + cP[12] * np.array(cP[13])
            netTotalForce += netFricForce+cP[9]*np.array(cP[7])


            # cP2 = contactPoints[gg]
            # print(
            #     "%s,pos On SJ: %s, pos on Box: %s, Contact Normal on Box towards SJ: %s, Contact Distance: %f, Normal Force: %f" % ( k, (cP2[5],), (cP2[6],), (cP2[7],), cP2[8], cP2[9]))

            # if gg>0:
            #     print("Multiple Contact Points")

        # Apply forces within distance of points
        zP = CF.ContactForce_ZeroPos(pressure=PressureValue) #get the distance where it is zero in mm, then scale by LScale and convert to m
        zP_scaled = zP*Lscale/1000
        contactPoints = p.getClosestPoints(SoftSupportInit, boxId, abs(zP_scaled[0]), SoftJawDict[k]["LinkIndex"], -1) #get all points within a distance of where the force goes to zero from the surface of the jaw
        numContactPoints=len(contactPoints)
        for gg in range(0,numContactPoints):

            cP = contactPoints[gg]
            #print("%s,pos On SJ: %s, pos on Box: %s, Contact Normal on Box towards SJ: %s, Contact Distance: %f, Normal Force: %f"%(k,(cP[5],),(cP[6],),(cP[7],),cP[8],cP[9]))

            contactNorm = np.array(cP[7])
            contactNormXY = np.array([contactNorm[0], contactNorm[1], 0]) / np.sqrt(contactNorm[0] ** 2 + contactNorm[1] ** 2) #ignore z component
            contactDist = np.sqrt(np.sum((np.array(cP[5])[0:1]-np.array(cP[6])[0:1]))**2)*np.sign(cP[8]) #positive for separation, negative for penetration
            if contactDist<=abs(zP_scaled):
                penDepth = -contactDist*1000/Lscale
                contactF = Lscale*CF.ContactForceCalc(PressureValue,penDepth ,zeroPos=zP)/numContactPoints  #assume the force is distributed evenly across the contact points ...
                contactStiff = CF.ContactStiffnessCalc(PressureValue, penDepth)*1000
                print(contactF)

                if NewApplyEF!=ApplyEF:
                    if False:
                        startIndex = index if len(SoftJawDict[k]["AppliedForceStart"][gg]) == 0 else SoftJawDict[k]["AppliedForceStart"][gg][0]
                        tstep=index-startIndex
                        SoftJawDict[k]["AppliedForceStart"][gg].append(index)
                        contactFDamped = contactF*min(tstep,numTsteps)/numTsteps
                        print(contactFDamped)
                        p.applyExternalForce(boxId, -1, contactNormXY*contactFDamped, cP[6], p.WORLD_FRAME)
                        p.applyExternalForce(SoftSupportInit, v["LinkIndex"], contactNormXY*-contactFDamped, cP[5], p.WORLD_FRAME)
                        print("On")


                    ## Change Stiffness
                    p.changeDynamics(SoftSupportInit, v["LinkIndex"],
                                     contactStiffness=contactStiff, contactDamping = v["Contact Damping"])
                    print(contactStiff)





        JointRF2 = p.getJointState(SoftSupportInit, v["LinkIndex"])
        #SoftJawDict[k]["JointForceTorque"].append(JointRF)

    #print("Net friction Force is: "+str(netFricForce))
    #print("Net Total Force is: " + str(netTotalForce))

    if False:
        netFricForce = 0
        netTotalForce = 0
        for k, v in SoftJawDict.items():
            DynInfo = p.getDynamicsInfo(SoftSupportInit, v["LinkIndex"])
            print(
                "%s properties: Mass %f, Lateral Friction %f, Rolling Friction %f, Spinning Friction %f, Contact Damping %f Contact Stiffness %f" % (
                k, DynInfo[0], DynInfo[1], DynInfo[6], DynInfo[7], DynInfo[8], DynInfo[9]))

            # Change stiffness and friction
            p.changeDynamics(SoftSupportInit, v["LinkIndex"], contactStiffness=v["Contact Stiffness"] * StiffScale,
                             contactDamping=v["Contact Damping"], lateralFriction=v["Lateral Friction"] * FricScale)

            # contactPoints = p.getContactPoints(SoftSupportInit, boxId, SoftJawDict[k]["LinkIndex"],-1)
            contactPoints = p.getContactPoints(boxId, SoftSupportInit, -1, SoftJawDict[k]["LinkIndex"])


            for gg in range(0, len(contactPoints)):

                cP = contactPoints[gg]

                netFricForce += cP[10] * np.array(cP[11]) + cP[12] * np.array(cP[13])
                netTotalForce += netFricForce + cP[9] * np.array(cP[7])

        print("Net friction Force (reversed) is: " + str(netFricForce))
        print("Net Total Force (reversed) is: " + str(netTotalForce))


    p.stepSimulation()
    index +=1



p.disconnect()



# p.resetSimulation()
# p.stopStateLogging(logId)

print("Finished")

fig, axs = pltv.subplots(len(SoftJawDict),1)
counter=0



print("Box Mass:%f"%MassOfBox)
print(switchOn)
colM = ['r-','b-','g-']



for k,v in SoftJawDict.items():

    ReactionJT = np.array(SoftJawDict[k]["JointForceTorque"])

    # SoftJawDict[k]["ContactNumber"].append(gg)
    # SoftJawDict[k]["NormalForce"].append(cp[9] * np.array(cP[3]))
    # SoftJawDict[k]["FrictionForce1"].append(cp[10] * np.array(cP[11]))
    # SoftJawDict[k]["FrictionForce2"].append(cp[12] * np.array(cP[13]))
    # SoftJawDict[k]["SpringForce"].append(cp[8] * v["Contact Stiffness"] * StiffScale)
    # SoftJawDict[k]["PenetrationDistance"].append(cp[8])
    DF = pd.DataFrame.from_dict({key: SoftJawDict[k][key] for key in ["ContactNumber","IndexContact","NormalForce","FrictionForce1","FrictionForce2","SpringForce","PenetrationDistance"]})

    ##split the columns with array elements to x, y and z
    multiComponentNames = ["NormalForce","FrictionForce1","FrictionForce2"]

    for vv in multiComponentNames:
        names = ["x", "y", "z"]
        ColNames = [vv+x for x in names]
        DF[ColNames]=pd.DataFrame(DF[vv].to_list(),index=DF.index)

    DF.to_csv(k+".csv")



    ## Plot Spring Force
    DF.plot("IndexContact","SpringForce")

    ## Plot the


    t = np.array(range(0,np.shape(ReactionJT)[0]))


    for i in range(0,3):
        axs[counter].plot(ReactionJT[:,i],colM[i],label=str(i))

    axs[counter].legend()
    counter +=1
pltv.show()
pltv.pause(1)