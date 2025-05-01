import pybullet as p
import time
import pybullet_data
import math
import pathlib
import warnings
import numpy as np
from pybullet_utils import bullet_client as bc
from pybullet_utils import urdfEditor as ed
import matplotlib.pyplot as plt
import re
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve

class GantrySimulation:
    """
    Class to instantiate the Gantry Simulation.  Currently using PyBullet

    Attributes
    ----------
    bulletClient    :   pyBullet client.
                        Can be instantiated in the calling code by:
                        pybullet_utils import bullet_client as bc
                        p0 = bc.BulletClient(connection_mode=pybullet.DIRECT)
                        see: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_utils/examples/multipleScenes.py

    lengthScale :   Scaling factor for all length dimensions in the simulation.  Default is 1

    massScale   :   Scaling factor for all mass dimensions in the simulation.  Default is 1.

    gantryId   :   object Id for the instantiated Gantry in the simulation

    simCounter  :   Increments as the simulation is stepped.

    timeStep    :   how long between simulation steps.  Defaults to 1/240 seconds which aligns with the PyBullet recommended
                    settings.

    Methods
    -------

    """
    def __init__(self, bulletClient: p = None, lengthScale: float = 1, massScale: float = 1, gravity_m_s2: float = -9.81,
                 timeStep = 1/240 ,
                 gantryURDFfile = str(pathlib.Path.cwd()/"Gantry\\envs\\URDF\\GantryAssembly_NoGrasper\\urdf\\GantryAssembly_NoGrasper.urdf"),
                 gantryURDFstartPos = [0,0,0], gantryURDFstartOrientation_quat = p.getQuaternionFromEuler([0,0,0]),
                 GantryLinkIndex_dict={"BasePositionIndex":12,"ZAxisBarIndex":1,"GantryHeadIndex":2,"CenterPlatformIndex":13},
                 grasperURDFfile: str = str("Grasper/SoftGrasper/URDF/SoftGrasperAssembly_SimplifiedTilt/urdf/SoftGrasperAssembly_SimplifiedTilt.urdf"),
                 MergeGrasperURDF: bool = True,GrasperType : str ="SoftGrasper"):
        """
        Constructor for the Gantry Simulation

        Parameters
        ----------
        bulletClient    :   pyBullet Client (optional)
            pyBullet Client.  If this is not passed, it will be initialized in the constructor

        lengthScale :   float (optional)
            Scaling factor for all length dimensions in the simulation.  Default is 1

        massScale   :   float (optional)
            Scaling factor for all mass dimensions in the simulation.  Default is 1

        gravity_m_s2    :   float (optional)
            Signed value for acceleration due to gravity.
            By default gravity acts along -ve z direction.  Default value is -9.81 m/s^2.  Expected units are m/s^2
            Value will be scaled by massScale and lengthScale

        timeStep    :   float (optional)
            Timestep for the simulation.  Default is 1/240 seconds

        gantryURDFfile  :   str (optional)
            location of the gantry URDF file and the associated .stl files.  Default is in the parent folder at
            Gantry\\envs\\URDF\\GantryAssembly_NoGrasper\\urdf\\GantryAssembly_NoGrasper.urdf

        gantryURDFstartPos  :   list of floats (optional)
            location of the base of the Gantry. [0,0,0] by default

        gantryURDFstartOrientation_quat  :   list of floats (optional)
            quaternion describing the orientation of the gantry.
            Use pyBullet.getQuaternionFromEuler([alpha,beta,gamma]) to translate from euler angles.
            Default is [0,0,0] in Euler.

        GantryLinkIndex_dict    :   dict (optional)
            Dict of the indices for the joints/links for the gantry.  Need to update as needed if the gantry design changes
            By Default:
            the z-axis aligns with movement of the bar ("ZAxisBarIndex").  Index is 1
            the y-axis aligns with movement of the base ("BasePositionIndex"). Index is 4
            the x-axis aligns with movement of the gantry head ("GantryHeadIndex").  Index is 2
            The center platform on which the object rests has an index of 5

            If you use the merge with grasper feature, then the link index will be different from above.

        grasperURDFfile         :   str (optional)
            File location of the URDF file to be used for loading the grasper

        MergeGrasperURDF        :   bool (optional)
            If true, then will call the pybullet merge URDF function to merge the urdfs of the grasper and the gantry.
            If false, then they will each be loaded as unconnected bodies.  Recommend to keep this as True (default)
            because it you load them as disjoint bodies and try to connect them with a fixed joint, it will be very loose,
            and not be "fixed".  The Merged urdf will be saved as "combined.urdf"

        GrasperType             :   str (optional)
            Currently only supported option is "SoftGrasper".  This will load the SoftGrasper Class.




        """
        self.bulletClient = bulletClient
        self.lengthScale = lengthScale
        self.massScale = massScale
        self.gravity = gravity_m_s2*self.massScale*self.lengthScale # default units is m/2^s (i.e. with massScale = 1 and lengthScale =1)
        self.timeStep = timeStep #seconds
        self.gantryURDFfile = gantryURDFfile
        self.gantryId = None #id for the gantry, to be set in setupGantry
        self.GUIcontrols = dict() #to be populated shortly
        self.maximumDataPoints = 60000 #maximum number of datapoints to collect before it starts rolling over
        self.GUIcontrolNum = 1 #incremented by 1 everytime the button on the GUI is pressed to pass control from the code to the GUI
        self.GantryLinkIndex_dict = GantryLinkIndex_dict #contains the indices of the different links in the gantry
        self.simCounter = 0


        self.jointForce_dict = {}  #to store joint forces and torques (not implemented currently)
        self.timelog = np.zeros(self.maximumDataPoints) #store the time point of the simulation in the time log.  rolls over when it reaches the maximum number of datapoints
        self.timelog_real = np.zeros(self.maximumDataPoints) #store the actual clock time of the simulation in the time log.  rolls over when it reaches the maximum number of datapoints
        self.LinkPosition_dict={} #to store link positions
        self.LinkOrientation_dict={} #to store link orientations
        self.ObjectPosition_dict={} #to store object positions (of the base link)
        self.ObjectOrientation_dict={} #to store object orientations (of the base link)
        self.GrasperURDF=grasperURDFfile
        self.Grasper=None #grasper object, e.g. SoftGrasper
        self.gantryLinkDict = {} #to store contact information



        if bulletClient is None:
            self.bulletClient = bc.BulletClient(connection_mode=p.GUI)
            self.bulletClient.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.bulletClient.setGravity(0,0,self.gravity) #z down is gravity
            #self.bulletClient.setPhysicsEngineParameter(contactERP=0.2)

        if MergeGrasperURDF == True:
            self.gantryURDFfile = self.MergeGrasperGantry()

        self.gantryId = self.setupGantry(gantryURDFstartPos,gantryURDFstartOrientation_quat) #import gantry
        self.planeId = self.bulletClient.loadURDF("plane.urdf")
        self.objects = {} #list of dictionaries containing all objects added to the world

        self.setupSim(GrasperID=None if MergeGrasperURDF == False else self.gantryId, GrasperType = GrasperType)

    def setupGantry(self,gantryURDFstartPos,gantryURDFstartOrientation_quat):
        """
        Load Gantry from URDF into the world

        Parameters
        ----------

        gantryURDFstartPos  :   float
            <x,y,z> start position

        gantryURDFstartOrientation_quat :   float
            quat for the orientation of the Gantry in the world

        Return
        ------
        gantryId    :   int
            ID of the gantry object
        """
        gantryId = self.bulletClient.loadURDF(self.gantryURDFfile,
                                                   [self.lengthScale*x for x in gantryURDFstartPos], gantryURDFstartOrientation_quat,
                                                   useFixedBase=True, globalScaling=self.lengthScale,
                                                   flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT | p.URDF_USE_MATERIAL_COLORS_FROM_MTL)

        # scale masses appropriately
        link_name_to_index = {self.bulletClient.getBodyInfo(gantryId)[0].decode('UTF-8'): -1, }
        for idv in range(self.bulletClient.getNumJoints(gantryId)):
            namev = self.bulletClient.getJointInfo(gantryId, idv)[12].decode('UTF-8')
            link_name_to_index[namev] = idv
            mass = self.bulletClient.getDynamicsInfo(gantryId, idv)[0]
            jointInfo = self.bulletClient.getJointInfo(gantryId, idv)
            self.bulletClient.changeDynamics(gantryId, idv, mass=mass * self.massScale)
            self.bulletClient.enableJointForceTorqueSensor(gantryId,idv,True) #enable force/torque sensing at the joints


        for i in range(self.bulletClient.getNumJoints(gantryId)):
            self.bulletClient.setJointMotorControl2(gantryId, i, p.POSITION_CONTROL, targetPosition=0,
                                                    force=500 * self.massScale * self.lengthScale)

        return(gantryId)


    def addGrasper(self,type="SoftGrasper",GrasperID=None):
        if type=="SoftGrasper":
            self.Grasper = SoftGrasper(bulletClient=self.bulletClient,lengthScale=self.lengthScale,massScale=self.massScale,GrasperID=GrasperID)



    def getForcesAndTorques(self):

        for k,v in self.objects.items():
            v.getContactForce(GantrySimObject=self, maxArraySize=self.maximumDataPoints, currentIndex=self.simCounter, gantryLinkDict=self.gantryLinkDict)



    def MergeGrasperGantry(self,parentLinkIndex=2,URDFname = "combined.urdf"): #see: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_utils/examples/combineUrdf.py
        p0 = bc.BulletClient(connection_mode=p.DIRECT)

        p1 = bc.BulletClient(connection_mode=p.DIRECT)

        # can also connect using different modes, GUI, SHARED_MEMORY, TCP, UDP, SHARED_MEMORY_SERVER, GUI_SERVER

        gantry = p1.loadURDF(self.gantryURDFfile)
        grasper = p0.loadURDF(self.GrasperURDF)

        #store joint limits
        JL = {}
        for k in range(0, p0.getNumJoints(gantry)):
            JointI = p0.getJointInfo(gantry,k)
            JL[JointI[1].decode() ] = [JointI[8],JointI[9]] #store lower and upper limits
        for k in range(0, p1.getNumJoints(grasper)):
            JointI = p1.getJointInfo(grasper, k)
            JL[JointI[1].decode() ] = [JointI[8],JointI[9]] #store lower and upper limits



        ed0 = ed.UrdfEditor()
        ed0.initializeFromBulletBody(gantry, p1._client)
        ed1 = ed.UrdfEditor()
        ed1.initializeFromBulletBody(grasper, p0._client)
        # ed1.saveUrdf("combined.urdf")

        parentLinkIndex = parentLinkIndex+1

        jointPivotXYZInParent = [-0.028, 0, 0]
        jointPivotRPYInParent = [-np.pi/2, 0, 0]

        jointPivotXYZInChild = [0, 0, 0]
        jointPivotRPYInChild = [0, 0, 0]

        newjoint = ed0.joinUrdf(ed1, parentLinkIndex, jointPivotXYZInParent, jointPivotRPYInParent,
                                jointPivotXYZInChild, jointPivotRPYInChild, p0._client, p1._client)
        newjoint.joint_type = p0.JOINT_FIXED

        for i in range(0,len(ed0.urdfJoints)): #update the joints
            jName = ed0.urdfJoints[i].joint_name
            if jName in JL:
                ed0.urdfJoints[i].joint_lower_limit  = JL[jName][0]
                ed0.urdfJoints[i].joint_upper_limit = JL[jName][1]

        ed0.saveUrdf(URDFname)



        p0.disconnect()
        p1.disconnect()

        return (URDFname)


    def addObjectsToSim(self,objectName,startPos=[0,0,0], startOrientation_quat=p.getQuaternionFromEuler([0,0,0]),
                        sizeScaling=1,sourceFile="",mass_kg=1):
        """
        Add object to the simulation.  Currently only able to add via URDF files.  See Solidworks URDF explorer to export URDF from an assembly.  http://wiki.ros.org/sw_urdf_exporter#:~:text=The%20SolidWorks%20to%20URDF%20exporter,and%20robots%20(urdf%20files).

        Parameters
        ----------
        objectName    :   str (required)
            Name to identify the object

        startPos    :   float (optional)
            Location of the origin of the object in the world frame.  This should be an unscaled value.
            Scaling will be calculated in this function using the lengthScale class variable. Default [0,0,0]

        startOrientation_quat   :   float (optional)
            Orientation of the body (quaternion).  Default [0,0,0,0]

        sizeScaling :   float (optional)
            different from GantrySimulation.lengthScale. This is an additional variable to change the scale of the body.
            Final scaling is <size of the object>*GantrySimulation.lengthScale*sizeScaling

        sourceFile  :   str (required)
            Name of the file(s) to be loaded that define the object.  Currently only .urdf files are supported.

        mass_kg :   float   (optional_
            Mass of the object.  Default is 1 kg.  This value should not be scaled when passed to this function.  Scaling
            by massScale happens in this function.

        """
        self.objects[objectName] = GantryObject(objectName, GantrySimObject = self, startPos=startPos, startOrientation_quat=startOrientation_quat,
                     lengthScale = self.lengthScale, massScale = self.massScale, sizeScaling = sizeScaling, sourceFile = sourceFile,mass_kg=mass_kg)


    def setupSim(self,GrasperID=None,GrasperType=None,contactNames = ["BasePlateJoint","ObjectBaseJoint","SJ1","SJ2","SJ3"]):
        """
        Setup GUI elements for simulation.

        """

        BasePositionId = self.bulletClient.addUserDebugParameter("BasePosition: y-axis", -0.2 * self.lengthScale, 0.2 * self.lengthScale, 0)
        ZAxisBarId = self.bulletClient.addUserDebugParameter("ZAxis: z-axis", -1 * self.lengthScale, 1 * self.lengthScale, 0)
        GantryHeadDebugId = self.bulletClient.addUserDebugParameter("GantryHead: x-axis", -0.2 * self.lengthScale, 0.2 * self.lengthScale,0)
        GUIcontrolId = p.addUserDebugParameter("Switch Programmatic/GUI Control", 1, 0, self.GUIcontrolNum)  # start at 1 (4th argument)
        StopSimId = p.addUserDebugParameter("Stop Experiment", 1, 0, 1)  # start at 1 (4th argument)
        self.GUIcontrols ={"BasePositionId":BasePositionId, "ZAxisBarId":ZAxisBarId,  "GantryHeadDebugId":GantryHeadDebugId,"GUIcontrolId":GUIcontrolId,"StopSimId":StopSimId}



        #add grasper and create fixed constraint
        if GrasperType is not None:
            self.addGrasper(type=GrasperType,GrasperID=GrasperID)

            #populate the dictionary of the links we want to get contact forces and friction from
            contactNames= contactNames#name of joints we want info for
            for k in range(0, self.bulletClient.getNumJoints(self.gantryId)):
                JointI = self.bulletClient.getJointInfo(self.gantryId, k)
                mass = self.bulletClient.getDynamicsInfo(self.gantryId, k)[0]
                self.bulletClient.changeDynamics(self.gantryId, k, mass=mass * self.massScale)
                DynInfo = list(self.bulletClient.getDynamicsInfo(self.gantryId, k))
                if JointI[1].decode() in contactNames:
                    self.gantryLinkDict[JointI[1].decode()] = k



    def CheckStopSim(self):
        """
        Stop the simulation based on if the "Stop Experiment" button is pressed.  Returns true if stopped, false otherwise

        """
        GUIcontrolTarget = self.bulletClient.readUserDebugParameter(self.GUIcontrols["StopSimId"])
        if GUIcontrolTarget != 1:
            self.bulletClient.disconnect()
            return(True)

        return(False)





    def stepSim(self,usePositionControl=False,logForcesAndTorques = True, GUI_override = False, x_gantryHead=None,y_BasePos=None,z_AxisBar=None,x_force=5000,y_force=5000,z_force=5000,maxVelocity_x=0.1,maxVelocity_y=0.1,maxVelocity_z = 0.1, GrasperArguments={}):
        """
        Step the simulation.

        Parameters
        ----------
        usePositionControl  :   bool (optional)
            If True, then stepSim will perform position control of the joints based on the arguments for x,y,z position
            and force if the GUI Control button on the GUI is an integer multiple of 2.  Otherwise, if the GUI/Programmatic
            control button on the GUI is toggled (i.e. GUI is an odd number) and this is True, it will read the values from
            the GUI slider and enforce the position control via these values.  Defaults to False.

        logForceAndTorques  :   bool (optional)
            If True, records forces,torques and contact forces (and link and object positions and orientation).  Default is True.

        GUI_override    :   bool (optional)
            If usePositionControl is True, then you can override the "Switch Programmatic/GUI Control" GUI button counter
            and just use Programmatic control.  If usePositionControl is False, this has no effect.

        x_gantryHead    :   float
            x position of the gantry (i.e. the gantry head where the 3D printer extruder was located).  Value should
            not be scaled (i.e. anticipating this value in m) when passing it in.  Scaling will be calculated
            in this function.  Default is None

        y_BasePos    :   float
            y position of the gantry (i.e. the movable base).  Value should not be scaled when passing it in
            (i.e. anticipating this value in m).  Scaling will be calculated in this function.   Default is None

        z_AxisBar    :   float
            z position of the gantry (i.e. the gantry head where the 3D printer extruder was located).  Value should
            not be scaled when passing it in (i.e. anticipating this value in m). Scailing will be calculated in this
            function.  Default is None

        x_force :   float
            maximum force applied to enforce constraint in the x-axis (expect in Newtons).  Appropriate scaling by lengthScale and
            massScale will happen in this function.  Default is 50

        y_force :   float
            maximum force applied to enforce constraint in the y-axis (expect in Newtons).  Appropriate scaling by lengthScale and
            massScale will happen in this function.  Default is 500

        z_force :   float
            maximum force applied to enforce constraint in the z-axis (expect in Newtons).  Appropriate scaling by lengthScale and
            massScale will happen in this function. Default is 500

        maxVelocity_x :   float
            maximum velocity of the x axis.  Default is 0.1

        y_force :   float
            maximum velocity of the y axis.  Default is 0.1

        z_force :   float
            maximum velocity of the z axis. Default is 0.03

        GrasperArguments : dict
            Dict of arguments of the grasper dynamics function



        """


        time.sleep(self.timeStep)

        if usePositionControl == True:
            BasePositionIndex = self.GantryLinkIndex_dict["BasePositionIndex"]
            ZAxisBarIndex = self.GantryLinkIndex_dict["ZAxisBarIndex"]
            GantryHeadIndex = self.GantryLinkIndex_dict["GantryHeadIndex"]

            BasePosTarget = self.bulletClient.readUserDebugParameter(self.GUIcontrols["BasePositionId"])
            ZAxisBarTarget = self.bulletClient.readUserDebugParameter(self.GUIcontrols["ZAxisBarId"])
            GantryHeadTarget = self.bulletClient.readUserDebugParameter(self.GUIcontrols["GantryHeadDebugId"])
            GUIcontrolTarget = self.bulletClient.readUserDebugParameter(self.GUIcontrols["GUIcontrolId"])



            if (GUIcontrolTarget%2 == 1 and GUI_override == False):  #everytime the "Switch Programmatic/GUI Control" GUI button is clicked, the counter is
                                            # incremented. If it is an odd number, then control of the gantry is based
                                            #on the GUI sliders.  If it is an even number, then control of the gantry is
                                            #based on the x,y,z force and position arguments passed to this function.
                print("GUI Control")
                x_gantryHead = GantryHeadTarget
                y_BasePos = BasePosTarget
                z_AxisBar = ZAxisBarTarget

                #Control Grasper via GUI
                if self.Grasper is not None:
                    GrasperGUIVals = self.Grasper.ReadInteractionButtons()

                    for k, v in self.objects.items():
                        self.Grasper.GrasperDynamics(
                            frictionCoefficient=GrasperGUIVals["FricCoefficient"],
                            PressureValue=GrasperGUIVals["PressureValue"],
                            TargetJawPosition=GrasperGUIVals["JawPosition"], MaxJawForce=20,
                            MaxVel=0.1, MaxVertForce=100,
                            TargetVertPosition=GrasperGUIVals["VerticalPosition"],
                            MaxVertVel=0.5, objectID=self.objects[k].objId)





            else:
                print("Programmatic Control")
                #Control Grasper via. programmatic control
                for k, v in self.objects.items():
                    self.Grasper.GrasperDynamics(**GrasperArguments,
                                                 objectID=self.objects[k].objId)

            self.GUIcontrolNum = GUIcontrolTarget
            self.bulletClient.setJointMotorControl2(self.gantryId,
                                    BasePositionIndex,
                                    p.POSITION_CONTROL,
                                    targetPosition=y_BasePos,
                                    force=y_force * self.massScale * self.lengthScale,
                                                    maxVelocity=maxVelocity_y*self.lengthScale)

            self.bulletClient.setJointMotorControl2(self.gantryId,
                                    ZAxisBarIndex,
                                    p.POSITION_CONTROL,
                                    targetPosition=z_AxisBar,
                                    force=z_force * self.massScale * self.lengthScale,
                                                    maxVelocity=maxVelocity_z*self.lengthScale)

            self.bulletClient.setJointMotorControl2(self.gantryId,
                                    GantryHeadIndex,
                                    p.POSITION_CONTROL,
                                    targetPosition=x_gantryHead,
                                    force=x_force * self.massScale * self.lengthScale,
                                                    maxVelocity=maxVelocity_x*self.lengthScale)



        if logForcesAndTorques == True:
            self.getForcesAndTorques() #log forces and torques
            self.getPositionsAndOrientations() #log position and orientation

        self.timelog[self.simCounter % self.maximumDataPoints] = self.timeStep * self.simCounter
        self.timelog_real[self.simCounter % self.maximumDataPoints] = time.time()

        self.bulletClient.stepSimulation() #step simulation
        self.simCounter = self.simCounter+1 #increment counter when you step the simulation

    def getPositionsAndOrientations(self):
        """
        Get link and object positions and orientations in the global world frame.  Orientations are quaternions.

        """
        for k,v in self.gantryLinkDict.items():  #iterate through each link that we want contact information for
            linkInfo = self.bulletClient.getLinkState(self.gantryId,  v) # get contact between base link of object and the link on the gantry
            self.LinkPosition_dict.setdefault(k,np.zeros([self.maximumDataPoints,3]))[self.simCounter%self.maximumDataPoints] = linkInfo[0]
            self.LinkOrientation_dict.setdefault(k, np.zeros([self.maximumDataPoints,4]))[self.simCounter%self.maximumDataPoints] = linkInfo[1]

        for k,v in self.objects.items():  #iterate through each link that we want contact information for
            linkInfo = self.bulletClient.getBasePositionAndOrientation(v.objId) # get contact between base link of object and the link on the gantry
            self.ObjectPosition_dict.setdefault(k,np.zeros([self.maximumDataPoints,3]))[self.simCounter%self.maximumDataPoints] = linkInfo[0]
            self.ObjectOrientation_dict.setdefault(k, np.zeros([self.maximumDataPoints,4]))[self.simCounter%self.maximumDataPoints] = linkInfo[1]




    def plotContactForces(self):
        # here we are creating sub plots
        fig, axs = plt.subplots( len(self.gantryLinkDict.keys()), len(self.objects), figsize=(10, 8))

        if axs.ndim == 1:
            axs = np.expand_dims(axs, -1) #deal with the case when there is only 1 object to enable indexing by two dimensions

        lines = []

        for lname, x in zip(self.gantryLinkDict.keys(), range(0, len(self.gantryLinkDict.keys()))): #iterate through added objects
            for ob, y in zip(self.objects.values(), range(0, len(self.objects))): #iterate through links in that object
                ax = axs[x, y]
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Force (N)')
                ax.set_title("Contact between " + ob.objectName + " and " + lname)
                ax.set_xlim(0, self.timeStep * self.maximumDataPoints)
                #idx = (self.simCounter - 1) % self.maximumDataPoints
                # updating data values
                xdata =self.timelog
                ydata = ob.ContactForce_dict[lname][:,:]

                lines.append(ax.plot(xdata,ydata[:,0],'ro',label='x component'))
                lines.append(ax.plot(xdata,ydata[:,1],'bo',label='y component'))
                lines.append(ax.plot(xdata,ydata[:,2],'go',label='z component'))
                ax.legend()
                axs[x, y] = ax

        plt.tight_layout()
        plt.show()
        plt.pause(10)



class SoftGrasper:

    def __init__(self,bulletClient=None,GrasperID=None,location=[0, 0, 0.52816],URDFpath = "URDF/SoftGrasperAssembly_SimplifiedTilt/urdf/SoftGrasperAssembly_SimplifiedTilt.urdf",lengthScale=1,massScale=1):
        self.bulletClient = bulletClient
        self.lengthScale = lengthScale
        self.massScale = massScale
        self.SoftJawDict={}
        self.LinkDict={}
        self.GrasperID = self.bulletClient.loadURDF(URDFpath, np.array(location)*self.lengthScale, globalScaling=self.lengthScale, useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_MATERIAL_COLORS_FROM_MTL) if GrasperID is None else GrasperID #either load grasperID separately and pass it in, or leave GrasperID as none and supply a URDF
        self.GrasperControls={} #dictionary to hold GUI controls for the grasper
        self.CF = self.ContactForce()



        #Change initial stiffness and damping, scale masses setup Dict for later use

        for k in range(0, self.bulletClient.getNumJoints(self.GrasperID)):
            JointI = self.bulletClient.getJointInfo(self.GrasperID, k)
            mass = self.bulletClient.getDynamicsInfo(self.GrasperID, k)[0]
            self.bulletClient.changeDynamics(self.GrasperID, k, mass=mass * self.massScale)
            DynInfo = list(self.bulletClient.getDynamicsInfo(self.GrasperID, k))
            self.LinkDict[JointI[1].decode()] = k

            if DynInfo[9] == -1:
                DynInfo[9] = 0.8

            if re.match("SJ\d", JointI[1].decode()) is not None:  # check if Joint Name is SJ
                posInfo = self.bulletClient.getLinkState(self.GrasperID, k)
                # SoftJaw = p.loadSoftBody("URDF/SoftGrasperAssembly_SimplifiedTilt/meshes/SoftJaw1__sf.obj", simFileName = "URDF/SoftGrasperAssembly_SimplifiedTilt/meshes/SoftJaw1.vtk", basePosition=posInfo[4],baseOrientation=posInfo[5], scale=Lscale, mass=0.1, useNeoHookean=1,
                #                         NeoHookeanMu=400, NeoHookeanLambda=6
                #
                #                         00, NeoHookeanDamping=0.1, useSelfCollision=1,
                #                         frictionCoeff=.5, collisionMargin=0.0001)
                # SoftJawArr.append(SoftJaw)
                DynInfo = list(self.bulletClient.getDynamicsInfo(self.GrasperID, k))

                if DynInfo[8] == -1:  #stiffness
                    DynInfo[8] = 1
                if DynInfo[9] == -1: #damping
                    DynInfo[9] = 0.8

                self.SoftJawDict[JointI[1].decode()] = {"LinkIndex": k, "Mass": DynInfo[0], "Lateral Friction": DynInfo[1],
                                                   "Rolling Friction": DynInfo[6], "Spinning Friction": DynInfo[7],
                                                   "Contact Damping": DynInfo[8], "Contact Stiffness": DynInfo[9],
                                                   "JointForceTorque": [],
                                                   "ContactNumber": [], "IndexContact": [], "NormalForce": [],
                                                   "FrictionForce1": [], "FrictionForce2": [],
                                                   "SpringForce": [], "PenetrationDistance": [],
                                                   "AppliedForceStart": {0: [], 1: [], 2: [], 3: []}}

            self.bulletClient.enableJointForceTorqueSensor(self.GrasperID, k, enableSensor=True)

        # Initialize Grasper Controls
        self.InteractionButtons()



    def InteractionButtons(self):

        self.GrasperControls["JawPosSlider"] = self.bulletClient.addUserDebugParameter("Jaw Position Slider", 0, 0.1*self.lengthScale, 0)
        self.GrasperControls["GrasperVertPosSlider"] = self.bulletClient.addUserDebugParameter("Grasper Vertical Position", -0.1*self.lengthScale, 0.1*self.lengthScale, 0)
        self.GrasperControls["ContactFrictionSlider"] = self.bulletClient.addUserDebugParameter("Contact Friction Coefficient (0 to 1)", 0, 1, 0.5)
        self.GrasperControls["PressureSlider"] = self.bulletClient.addUserDebugParameter("Pressure (PSI)", 0, 2.5, 0)
        return(self.GrasperControls)

    def ReadInteractionButtons(self):
        retDict = {}

        retDict["FricCoefficient"] = 10 ** (self.bulletClient.readUserDebugParameter(
            self.GrasperControls["ContactFrictionSlider"]))  # convert from log scale
        retDict["PressureValue"] = self.bulletClient.readUserDebugParameter(self.GrasperControls["PressureSlider"]) #read pressure value
        retDict["VerticalPosition"] = self.bulletClient.readUserDebugParameter(
            self.GrasperControls["GrasperVertPosSlider"]) #vertical position
        retDict["JawPosition"] = self.bulletClient.readUserDebugParameter(
            self.GrasperControls["JawPosSlider"])  # position of jaws

        return(retDict)

    def GrasperDynamics(self,frictionCoefficient=None,PressureValue=None,TargetJawPosition=0,MaxJawForce=None,MaxVel=None,MaxVertForce=None,TargetVertPosition=None,MaxVertVel=None,objectID=None):

        for k in range(0, self.bulletClient.getNumJoints(self.GrasperID)):
            JointI = self.bulletClient.getJointInfo(self.GrasperID, k)
            if re.match("LinearJaw\d", JointI[1].decode()) is not None:  # check if Joint Name is SJ
                self.bulletClient.setJointMotorControl2(self.GrasperID,
                                        k,
                                        p.POSITION_CONTROL,
                                        targetPosition=TargetJawPosition*self.lengthScale,
                                        force=MaxJawForce*self.lengthScale*self.massScale,
                                        maxVelocity=MaxVel*self.lengthScale)
            if re.match("BCC", JointI[1].decode()) is not None:  # check if Joint Name is SJ
                self.bulletClient.setJointMotorControl2(self.GrasperID,
                                        k,
                                        p.POSITION_CONTROL,
                                        targetPosition=TargetVertPosition*self.lengthScale,
                                        force=MaxVertForce*self.lengthScale*self.massScale,
                                        maxVelocity=MaxVertVel*self.lengthScale)

        for k, v in self.SoftJawDict.items():
            DynInfo = self.bulletClient.getDynamicsInfo(self.GrasperID, v["LinkIndex"])
            # print("%s properties: Mass %f, Lateral Friction %f, Rolling Friction %f, Spinning Friction %f, Contact Damping %f Contact Stiffness %f"%(k,DynInfo[0],DynInfo[1],DynInfo[6],DynInfo[7],DynInfo[8],DynInfo[9]))

            # Change stiffness and friction
            self.bulletClient.changeDynamics(self.GrasperID, v["LinkIndex"],  lateralFriction=frictionCoefficient)

            zP = self.CF.ContactForce_ZeroPos(pressure=PressureValue)  # get the distance where it is zero in mm, then scale by LScale and convert to m
            zP_scaled = zP * self.lengthScale / 1000
            contactPoints = self.bulletClient.getClosestPoints(self.GrasperID, objectID, abs(zP_scaled[0]), self.SoftJawDict[k]["LinkIndex"],
                                               -1)  # get all points within a distance of where the force goes to zero from the surface of the jaw
            numContactPoints = len(contactPoints)
            for gg in range(0, numContactPoints):

                cP = contactPoints[gg]
                # print("%s,pos On SJ: %s, pos on Box: %s, Contact Normal on Box towards SJ: %s, Contact Distance: %f, Normal Force: %f"%(k,(cP[5],),(cP[6],),(cP[7],),cP[8],cP[9]))
                contactDist = np.sqrt(np.sum((np.array(cP[5])[0:1] - np.array(cP[6])[0:1])) ** 2) * np.sign(
                    cP[8])  # positive for separation, negative for penetration
                if contactDist <= abs(zP_scaled):
                    penDepth = -contactDist * 1000 / self.lengthScale
                    contactF = self.lengthScale * self.CF.ContactForceCalc(PressureValue, penDepth,
                                                            zeroPos=zP) / numContactPoints  # assume the force is distributed evenly across the contact points ...
                    contactStiff = self.CF.ContactStiffnessCalc(PressureValue, penDepth) * 1000


                    ## Change Stiffness
                    self.bulletClient.changeDynamics(self.GrasperID, v["LinkIndex"],
                                     contactStiffness=contactStiff, contactDamping=v["Contact Damping"])
                    print("Contact Stiffness: "+str(contactStiff))












    class ContactForce:
        def __init__(self, coefficients=[0.0416, 0.1791, 0.8912, 5.4641, 1.0647, 0.5050]):
            self.coefficients = coefficients
            self.ZeroDistance = 0

        '''
        Contact Force:
        pressure: value in gauge psi
        displacement: value in mm.  convention is 0 is the surface of the soft jaw, +ve values indicate compression into the jaw, -ve values indicate distances above the jaw
        '''

        def ContactForceFunc(self, pressure=0, displacement=0, coefficients=None):
            b = self.coefficients if coefficients is None else coefficients
            x = displacement
            P = pressure
            F = ((b[5] * P ** b[4] + b[0]) * ((b[1] ** (2 * P)) * x ** 2 + (b[2] ** P) * x + b[3]))
            return F

        def ContactForce_ZeroPos(self, pressure=0, coefficients=None):
            b = self.coefficients if coefficients is None else coefficients
            F_P = lambda x: self.ContactForceFunc(pressure=pressure, displacement=x, coefficients=b)
            zeroloc = fsolve(F_P, 0)
            self.ZeroDistance = zeroloc
            return (zeroloc)

        def ContactForceCalc(self, pressure, displacement, coefficients=None, zeroPos=None):
            b = self.coefficients if coefficients is None else coefficients
            F_val = self.ContactForceFunc(pressure=pressure, displacement=displacement, coefficients=b)
            zPos = self.ContactForce_ZeroPos(pressure=pressure, coefficients=b) if zeroPos is None else zeroPos
            F_val = F_val if (displacement > zPos or zPos > 0) else 0
            return F_val

        def ContactStiffnessCalc(self, pressure, displacement, coefficients=None):  # Returns in N/mm
            b = self.coefficients if coefficients is None else coefficients
            StiffVal = (b[0] + (pressure ** b[4]) * b[5]) * (
                        b[2] ** pressure + 2 * (b[1] ** (2 * pressure)) * displacement)
            return StiffVal





class GantryObject:

    def __init__(self,objectName,GantrySimObject=None,startPos=[0,0,0],startOrientation_quat=p.getQuaternionFromEuler([0,0,0]), lengthScale=1,
                 massScale=1, sizeScaling=1, sourceFile = "",mass_kg=1):
        self.startPos = startPos
        self.startOrientation = startOrientation_quat
        self.objectName = objectName
        self.objId =None #to be initialized later
        self.lengthScale = lengthScale
        self.massScale = massScale
        self.sourceFile = sourceFile
        self.ObjectType = pathlib.Path(sourceFile).suffix #depends on the extension of the file.  Should either be URDF or .stl
        self.sizeScaling=sizeScaling
        self.ContactForce_dict = {}  # to store contact forces between gantry and objects.  Structure will be <objectName>:[[normal contact force x, normal contact force y, normal contact force z]]
        self.FrictionForce_dict={} #to store the frictional forces on the object

        if str.lower(self.ObjectType) not in [str.lower(".urdf"),str.lower(".stl")]:
            warnings("Gantry Objection Initialization: the extension is neither .urdf or .stl.")

        if str.lower(self.ObjectType) == str.lower(".urdf"):
            if type(GantrySimObject) is GantrySimulation:
                if type(GantrySimObject.bulletClient) is bc.BulletClient:
                    self.objId = GantrySimObject.bulletClient.loadURDF(sourceFile, [self.lengthScale*x for x in self.startPos], self.startOrientation,
                                       useFixedBase=True, globalScaling=self.lengthScale*self.sizeScaling,
                                       flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_MATERIAL_COLORS_FROM_MTL)

                    GantrySimObject.bulletClient.changeDynamics(self.objId, -1, mass=mass_kg * massScale)

    def getContactForce(self, GantrySimObject =None, maxArraySize = 10000, currentIndex = 0, gantryLinkDict={}):
        #gantryLinkDict should be like {"BasePositionIndex":6}
        #[x[9] for x in gS.bulletClient.getContactPoints(gS.gantryId, gS.objects["PickupCube"].objId, 7, -1)]
        #g.setdefault("kk",np.zeros(3))[2]=3
        gS = GantrySimObject

        for k,v in gantryLinkDict.items():  #iterate through each link that we want contact information for
            ContactPointInfo = gS.bulletClient.getContactPoints(gS.gantryId, self.objId, v, -1) # get contact between base link of object and the link on the gantry
            TotalContactForceVector = np.array([0,0,0]) #x,y,z components of normal contact force between bodies
            TotalFrictionForceVector = np.array([0,0,0]) #lateral friction force between bodies
            if len(ContactPointInfo) != 0:
                TotalContactForceVector = sum(np.array([np.array(x[7])*x[9] for x in ContactPointInfo])) #7th element is the contact normal.  9th elements is the normal contact force.  Sum the vector of forces across all contact points
                TotalFrictionForceVector = sum(np.array([cP[10] * np.array(cP[11]) + cP[12] * np.array(cP[13]) for cP in ContactPointInfo]))
                print(k+":"+str(TotalContactForceVector))
            self.ContactForce_dict.setdefault(k,np.zeros([maxArraySize,3]))[currentIndex%maxArraySize] = TotalContactForceVector
            self.FrictionForce_dict.setdefault(k,np.zeros([maxArraySize,3]))[currentIndex%maxArraySize] = TotalFrictionForceVector












