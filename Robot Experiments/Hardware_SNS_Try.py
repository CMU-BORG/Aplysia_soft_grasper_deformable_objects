import asyncio
import socketio

import time

from GUI.EmbeddedSystems.SoftGrasper.SoftGrasper import PortActions
from GUI.EmbeddedSystems.SoftGrasper.SoftGrasper import SoftGrasper
from GUI.EmbeddedSystems.Gantry.GantryController import Gantry as GantryController
import GUI.EmbeddedSystems.JoyCon.JoyCon as JC
from GUI.EmbeddedSystems.SNS.SNScontroller import SNScontroller
from GUI.EmbeddedSystems.Support.Structures import GrasperContactForce,Point
from enum import Enum

import logging
from datetime import datetime
import sys
from pathlib import Path

##### Set up logging ####
l_date = datetime.now().strftime("_%d_%m_%Y_%H_%M_%S")
loggerR = logging.getLogger(__name__)
fname = Path(__file__).parents[0].joinpath("datalogs","Hardware"+l_date+".txt")

fh = logging.FileHandler(fname) #file handler
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout) #stream handler
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh.setFormatter(formatter)
ch.setFormatter(formatter)
loggerR.setLevel(logging.DEBUG)

# add the handlers to the logger
loggerR.addHandler(fh)
loggerR.addHandler(ch)



#### Set up logging data logger
datalogger = logging.getLogger(__name__+"_datalogs")

fname = Path(__file__).parents[0].joinpath("datalogs","HardwareDatalog"+l_date+".csv")

fh2 = logging.FileHandler(fname) #file handler
fh2.setLevel(logging.DEBUG)

ch2 = logging.StreamHandler(sys.stdout) #stream handler
ch2.setLevel(logging.ERROR)

formatter2 = logging.Formatter('%(asctime)s,%(name)s,%(levelname)s,%(message)s')

fh2.setFormatter(formatter2)
ch2.setFormatter(formatter2)

datalogger.setLevel(logging.DEBUG)

# add the handlers to the logger
datalogger.addHandler(fh2)
datalogger.addHandler(ch2)




SG  = None #soft grasper
GC = None #gantry controller
jcSG = None #joystick
SNSc = None #SNS controller

useSG = True #set to true to use the soft grasper
useGC = True #set to true to use the gantry
useJoy = True #set to true to use the joystick
usejcSG = useJoy and useSG and useGC #set to true to use the joystick with the soft grasper and the gantry controller
useSNS = True #set to true if you want to use the SNS to control

buttonVal = None #read from joystick
AxesPos = None #joystick axes position

fragileThreshold = 0.4 #scale of 0 to 1.  Forces above this are considered broken for fragile objects

max_z_height = -0.184
SNS_object_pos_m = [0,0,max_z_height]
SNS_target_pos_m = [0,-0.25,max_z_height]
z_offset = 0

pressureThreshold = [0.05,0.05,0.05] #threshold above which to trigger the rumble, psi
pressureScaling = 1 #100% pressure value for fumble.  In psi. formula is (max change in pressure - pressurethreshold)/PressureScaling


class itemStatus(str,Enum):
    notstarted = 'notstarted'
    inprogress = 'inprogress'
    fail = 'fail'
    success = 'success'
    broken = 'broken'

class itemClass(Enum):
    fixed= 'fixed'
    breakable = 'breakable'
    graspable = 'graspable'
    notspecified = 'notspecified'

class itemTrack:
    def __init__(self,objectID, status = itemStatus.notstarted,itemClass=itemClass.graspable):
        self.objectID = objectID
        self.status = status #Not started, inprogress, fail, broken, success
        self.itemClass = itemClass #fixed, breakable, graspable

#object 1 is top left, object 9 is bottom right.
ObjectVal = {"1": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
             "2": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
             "3": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
             "4": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
             "5": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
             "6": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
             "7": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
             "8": itemTrack(1, itemStatus.notstarted, itemClass.notspecified),
             "9": itemTrack(1, itemStatus.notstarted, itemClass.notspecified)
             }

localHostName = 'http://localhost:8000'

updatedSoftGrasper = False


#loop = asyncio.get_event_loop()
sio = socketio.AsyncClient()
start_timer = None




async def send_ping():
    global start_timer, loggerR
    loggerR.debug('Sending ping')

    start_timer = time.time()
    await sio.emit('my event','New Data')

async def send_Start():
    global start_timer, loggerR
    loggerR.debug('Sending Start')

    start_timer = time.time()
    await sio.emit('start event','Ready to Start')

@sio.event
async def connect():
    global loggerR
    loggerR.debug('connected to socket')
    await send_Start()

@sio.on('gantry position commands')
async def gantryPosition(data):

    global GC, loggerR, useGC, useSNS, SNS_object_pos_m, SNS_target_pos_m, max_z_height
    loggerR.info('Received gantry position command')
    loggerR.debug(data)
    if useGC == True:
        GC.goalPos = [data['x']-20, data['y']-20, GC.goalPos[2]] #x and y positions come from screen, z position is kept from the joystick
    else:
        loggerR.debug("Gantry disabled")
    loggerR.debug("GoalPos changed in Gantry Position: " + ','.join([str(x) for x in GC.goalPos]))

    #if using the SNS control, update the object position x, y to be the goal pos.  the Z is a fixed height of -0.189 m
    if useSNS == True:
        SNS_fixed_height_z = max_z_height #height of object in meters relative to start position. z height offset is 0.2 m.
        SNS_object_pos_m = [GC.goalPos[0]/1000, GC.goalPos[1]/1000, SNS_fixed_height_z]
        SNS_target_pos_m = [0.09, 0.05, SNS_fixed_height_z]
        loggerR.info('Commanded object pos in m: %f, %f %f'%(*SNS_object_pos_m,))
        loggerR.debug("Modified the SNS object and target positions")

@sio.on('soft grasper commands')
async def softGrasperCommands(data):

    global SG, updatedSoftGrasper, loggerR, useSG
    loggerR.info('Received soft grasper power command')

    closure_mm = data['grasper_l']*20/100
    loggerR.debug("Change in radius in mm: "+str(closure_mm))

    if useSG == True:
        SG.commandedPosition["ClosureDistance_mm"] = closure_mm
        SG.commandedPosition["Jaw1_psi"] = 0
        SG.commandedPosition["Jaw2_psi"] = 0
        SG.commandedPosition["Jaw3_psi"] = 0

    else:
        loggerR.debug("Soft grasper disabled")

    updatedSoftGrasper = True

    #SG.AbsoluteMove(closureIncrement_mm=closureIncrement_mm, jawIncrement_psi=[0, 0, 0]) #setup internal variables to actuate at MoveGrasper
    #SG.MoveGrasper()  # actuate soft grasper


@sio.on('item-event_emit')
def handle_item_event_hardware(data):

    global loggerR, ObjectVal

    n = data['n']
    typ = data['typ']
    attempt = data['att']
    test = data['test']
    loggerR.info('Item event: item %s is %s.  Attempt %s, test %s)'%(n,typ,attempt,test) )
    ObjectVal[str(int(n)+1)].status = itemStatus[typ]


@sio.on('TestInfo')
def handle_TestInfo(user_data, trial_data):

    global loggerR, ObjectVal
    id_code = user_data['id_code']
    age = user_data['age']
    gender = user_data['gender']
    types = user_data['item_types']

    test = trial_data['test']

    num_attempts = trial_data['num_attempts']

    loggerR.info("ID code %s, age: %s, gender: %s, types: %s, test: %s, num_attempts: %s"%(id_code, age,
                                                                                           gender, types,
                                                                                           test, num_attempts))

    for k,v in enumerate(types):
        v = v.lower()
        if v in [str(x.name) for x in itemClass]:
            ObjectVal[str(k+1)].itemClass = itemClass[v] #assign object Type

        else:
            ObjectVal[str(k + 1)].itemClass = itemClass.notspecified





async def HardwareInitialize():
    global SG, GC, jcSG, loggerR, useSG, useGC, usejcSG, SNSc, z_offset

    if useSG == True:
        loggerR.info('Initializing Soft Grasper...')
        SG = SoftGrasper(COM_Port='COM7', BaudRate=460800, timeout=1, controllerProfile="New") #initialize soft grasper
        loggerR.info('Finished initializing Soft Grasper')

    if useGC == True:
        loggerR.info('Initializing Gantry...')
        GC = GantryController(comport = "COM4")#, homeSystem = False,initPos=[0,0,0]  #initialize gantry controller
        loggerR.info('Finished initializing Gantry!')

    if usejcSG == True:
        loggerR.info('Initializing Joystick...')
        jcSG = JC.Joy_SoftGrasper(SGa=SG, GantryS=GC) #initialize joystick control of soft grasper and gantry controller
        loggerR.info('Finished initializing Joystick!')


    if useSNS == True:
        loggerR.info('Initializing SNS controller')
        SNSc = SNScontroller()
        loggerR.info('Finished initializing the SNS.')

    loggerR.info('Finished Initialization')

@sio.on('program_loop')
async def program_loop():
    global SG, GC, jcSG, updatedSoftGrasper, loggerR, datalogger, buttonVal, AxesPos, useSG, useGC, usejcSG, SNSc, SNS_object_pos_m, SNS_target_pos_m, pressureScaling, pressureThreshold, max_z_height

    #await sio.emit('gantry position commands', 'Gantry pos')
    #await sio.emit('soft grasper commands', 'Soft Grasper Commands')
    try:
        while(True):

            if usejcSG == True:
                [buttonVal, AxesPos] = jcSG.eventLoop()  # run event loop to determine what values and axes to execute
                jcSG.ExecuteButtonFunctions(buttonVal,
                                            AxesPos)  # execute Button functions defined by the button values and axes positions

                if jcSG.SNS_control == False or useSNS == False:
                    posInc,feedrate_mmps = jcSG.calcPositionIncrement(AxesPos[0], AxesPos[1])  # get the joystick position for x, y and z, corrected for the flip on the x axis
                    GC.calculateIncrementalMove(*posInc) #will set the GC.goalPos with the appropriate increments

                #loggerR.info("Current Z: %f"%(GC.PositionArray["z"][-1]))
                # SNS Control
                if useSNS == True and jcSG.SNS_control == False:
                    SNSc = SNScontroller()
                    #logger.info('Reset controller')


                elif jcSG.SNS_control == True and useSNS == True: # if the user presses the SR button and you want to have the option to switch to SNS, then use SNS to calculate the next commands

                    loggerR.info('Inside SNS control')

                    curPos = GC.getPosition()  # get current position, in meters relative to offset


                    if SNSc.first_attempt == True:
                        z_offset = curPos.z
                        # will adjust height relative to object position

                        loggerR.info("Z offset %f"%z_offset)
                        object_position_list = [0, 0, SNS_object_pos_m[2] - z_offset]  # [0, 0, -0.185] #calculate move relative to the object position.
                        # [-0.15, -0.15, -0.190]

                        target_position_list = [SNS_target_pos_m[0] - SNS_object_pos_m[0],
                                                SNS_target_pos_m[1] - SNS_object_pos_m[1],
                                                SNS_target_pos_m[2] - z_offset]



                        SNSc.first_attempt = False


                    curPos_orig = curPos
                    curPos = Point(-SNS_object_pos_m[0]+curPos.x, -SNS_object_pos_m[1]+curPos.y,
                                   -z_offset + curPos.z) #get where the current position is relative to where the object is



                    grasperPosition = Point(curPos.x, curPos.y, curPos.z)  # convert to meters
                    loggerR.info('Target position in m: %f %f %f' % (*target_position_list,))
                    loggerR.info('GrasperPos in m:%f %f %f' % (*grasperPosition,))
                    loggerR.info('ObjectPos in m:%f %f %f' % (*object_position_list,))

                    grasperThreshold = [0.05, 0.05, 0.05]
                    grasperContact = [(x - pressureThreshold[i]) * 30 if x >= grasperThreshold[i] else 0 for (i, x) in
                                      enumerate(SG.changeInPressure)]

                    grasperContact = GrasperContactForce(*grasperContact)

                    commandPosition_m, JawRadialPos_m = SNSc.SNS_forward(grasperPos_m=grasperPosition,
                                                                         grasperContact=grasperContact,
                                                                         objectPos_m=Point(*object_position_list),
                                                                         targetPos_m=Point(*target_position_list),useRealGantry=True)

                    loggerR.info('Jaw radial pos in m:%f'%(JawRadialPos_m))
                    loggerR.info('Command Position: %f %f %f'%(*list(commandPosition_m),))
                    # command position is absolute move in m relative to the offset.
                    commandPosition_m = Point(SNS_object_pos_m[0] + commandPosition_m.x,
                                              SNS_object_pos_m[1] + commandPosition_m.y,
                                              z_offset+commandPosition_m.z)#calculate the move relative to the object position

                    GC.goalPos = [x * 1000 for x in list(commandPosition_m)]

                    loggerR.info('SNS commanded goal position (mm):%f,%f,%f'%(*GC.goalPos,))
                    loggerR.debug('SNS commanded change in radius (mm):%f' % (JawRadialPos_m*1000))
                    loggerR.debug(','.join([k + ":" + str(v) for k, v in SNSc.neuronset.items()]))

                    if (abs(curPos_orig.z-SNS_target_pos_m[2])>0.03 and
                            (SNSc.neuronset["move_to_grasp"]>=20 or SNSc.neuronset["move_to_pre_grasp"]>=20)): #don't inflate when far from the object in z
                        JawRadialPos_m = 0

                    SG.commandedPosition["ClosureDistance_mm"] = min(JawRadialPos_m * 1000,14) #limit the radial position change to prevent overinflation

                    if SNSc.lift_after_release_done == True:
                        SNSc = SNScontroller() #reinitialize the SNS controller
                        jcSG.SNS_control = False #reset to false to give control back to the user
                        loggerR.info('Reset the SNS controller after lift after release complete')

                    loggerR.info('Number of grasp attempts %i' % SNSc.num_grasp_attempts)
                    if SNSc.num_grasp_attempts >=1 or SNSc.lift_after_grasp_done == True:

                        if (SNSc.num_grasp_attempts>=1
                                and
                                (SNSc.neuronset["move_to_grasp"]>=20 or SNSc.neuronset["move_to_pre_grasp"]>=20)): #if about to attempt a regrasp
                            GC.goalPos = [curPos_orig.x*1000, curPos_orig.y*1000, curPos_orig.z*1000+70] #move object up
                            loggerR.info('Failed grasp, exceeded number of attempts')
                            SNSc.lift_after_grasp_done = True #set to true to trigger the next statement
                            SG.commandedPosition["ClosureDistance_mm"] = 0


                        if SNSc.num_grasp_attempts>=1 and SNSc.lift_after_grasp_done == True:
                            jcSG.SNS_control = False  # reset to false to give control back to the user

                            SNSc = SNScontroller()
                            loggerR.info('Reset the SNS controller after lift after grasp complete')



                GC.goalPos=Point(GC.goalPos[0],GC.goalPos[1],max(GC.goalPos[2],max_z_height*1000))

                GC.setXYZ_Position(*GC.goalPos,feedrate_mmps*60)  # absolute move of the gantry

            if useSG == True:
                SG.MoveGrasper()  # jcRG.executeButtonFunctions should update SGa with the the pressures, this command sends the appropriate commands to the grasper over serial


                if updatedSoftGrasper == True:
                    loggerR.debug("UPDATED Soft Grasper has happened")
                    updatedSoftGrasper = False

                # Rumble feedback based on pressure change
                rumbleValue = calculateRumble(pressureThreshold,pressureScaling)

            if usejcSG == True:
                jcSG.rumbleFeedback(rumbleValue, rumbleValue, 1000)

                await sio.emit('set-contact-force-soft_hardware', rumbleValue*100)
                await sio.emit('set-gantry-marker_hardware', {'x': GC.PositionArray['x'][-1],'y': GC.PositionArray['y'][-1]})


            await asyncio.sleep(0.001) #allow other tasks to run
            loggerR.debug('ProgramLoop')
            datalogFcn()


    except KeyboardInterrupt:
        loggerR.debug('Exiting loop')
        return

def calculateRumble(pressureThreshold = [0.2,0.2,0.2],pressureScaling=1):
    global SG, ObjectVal, fragileThreshold, loggerR
    #pressureThreshold:  change in pressure threshold in psi above which to register changes in pressure
    rumbleValue_arr = [min((x - pressureThreshold[i]) / pressureScaling, 1) if x >= pressureThreshold[i] else 0 for (i, x) in
                   enumerate(SG.changeInPressure)] #was divide by 1.75, changed to 1

    rumbleValue = max(rumbleValue_arr)
    # if the current attempt is on a breakable item, check to see if the rumble value is greater than the threshold

    idx = [k for k,x in ObjectVal.items() if x.status == itemStatus.inprogress] #find the key which in progress

    try:
        if len(idx) > 0:
            idx = idx[0]
            if ObjectVal[idx].itemClass == itemClass.breakable and rumbleValue > fragileThreshold:
                rumbleValue = 0
                loggerR.warning('Force exceeded on item %s.  Threshold %f, rumble value %f'%
                                (idx, fragileThreshold, rumbleValue))


    except Exception as e:
        loggerR.error(e)
        loggerR.error('Error during fragile object check')
        rumbleValue = 0

    return (rumbleValue)

def datalogFcn():
    global SG, GC, jcSG,  datalogger, buttonVal, AxesPos, useSG, useGC, usejcSG

    # should be x, y and z position of gantry, pressure in closure muscle, pressure in jaw 1, pressure in jaw 2, pressure in jaw3, button 1 status....button 11 status, axes 0 joystick, axes 1 joystick,  object 1 status .... object 9 status
    ds = ''

    if useGC == True:
        GCvals = [x[-1] if len(x)>0 else 1000 for k,x in GC.PositionArray.items()]
        ds="%f , %f , %f,"%(*GCvals[1:4],)

    if useSG == True:
        SGPressurevals = [SG.PressureArray[i][-1] if len(SG.PressureArray[i])>0 else 1000 for i in range(0,SG.numPorts)  ]
        ds = ds + "%f , %f , %f , %f,"%(SGPressurevals[SG.closureMuscle_idx],
                                       SGPressurevals[SG.JawPos[0]],
                                       SGPressurevals[SG.JawPos[1]],
                                       SGPressurevals[SG.JawPos[2]])

    if usejcSG == True:
        ds = ds +"%i , %i , %i , %i , %i , %i , %i , %i , %i , %i , %i ,  %f , %f," % (*list(buttonVal.values()),*AxesPos)


    ds = ds + "%s , %s , %s , %s , %s , %s , %s , %s , %s"% ( *[x.status.name for k,x in ObjectVal.items()], )



    datalogger.info(ds)





async def start_Program():
    await HardwareInitialize()
    loggerR.info('Triggering connection to ' + str(localHostName))
    await sio.connect(localHostName)
    await program_loop()
    await sio.wait()





if __name__ == '__main__':
    try:
        asyncio.run(start_Program())
    except KeyboardInterrupt:
        "Exiting Program.  Thank you."

    # executor = ProcessPoolExecutor(2) #https://stackoverflow.com/questions/29269370/how-to-properly-create-and-run-concurrent-tasks-using-pythons-asyncio-module
    # loop = asyncio.new_event_loop()
    # boo = loop.run_in_executor(executor, say_boo)
    # baa = loop.run_in_executor(executor, say_baa)
    #
    # loop.run_forever()