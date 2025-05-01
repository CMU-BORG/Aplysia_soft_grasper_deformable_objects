import asyncio
import socketio

import time

from EmbeddedSystems.SoftGrasper.SoftGrasper import PortActions
from EmbeddedSystems.SoftGrasper.SoftGrasper import SoftGrasper
from EmbeddedSystems.Gantry.GantryController import Gantry as GantryController
import EmbeddedSystems.JoyCon.JoyCon as JC

SG  = None #soft grasper
GC = None #gantry controller
jcSG = None #joystick

updatedSoftGrasper = False


#loop = asyncio.get_event_loop()
sio = socketio.AsyncClient()
start_timer = None


async def send_ping():
    print('Sending ping')
    global start_timer
    start_timer = time.time()
    await sio.emit('my event','New Data')

async def send_Start():
    print('Sending Start')
    global start_timer
    start_timer = time.time()
    await sio.emit('start event','Ready to Start')

@sio.event
async def connect():
    print('connected to socket')
    await send_Start()


@sio.on('SendInfo')
async def pong_from_server(data):
    global start_timer
    latency = time.time() - start_timer
    print('latency is {0:.2f} ms'.format(latency * 1000))
    print(data)
    await sio.sleep(1)
    if sio.connected:
        print('Ping')
        await send_ping()

@sio.on('gantry position commands')
async def gantryPosition(data):
    print('gantry')
    global GC
    latency = time.time()
    print('Time is {0:.2f} ms'.format(latency * 1000))
    print(data)
    GC.goalPos = [data['x'], data['y'], GC.goalPos[2]] #x and y positions come from screen, z position is kept from the joystick
    print("GoalPos changed in Gantry Position: " + ','.join([str(x) for x in GC.goalPos]))

@sio.on('soft grasper commands')
async def softGrasperCommands(data):
    print('softGrasper')
    global SG, updatedSoftGrasper

    closure_mm = data['grasper_l']*20/100
    print(closure_mm)
    SG.commandedPosition["ClosureDistance_mm"] = closure_mm
    SG.commandedPosition["Jaw1_psi"] = 0
    SG.commandedPosition["Jaw2_psi"] = 0
    SG.commandedPosition["Jaw3_psi"] = 0

    updatedSoftGrasper = True

    #SG.AbsoluteMove(closureIncrement_mm=closureIncrement_mm, jawIncrement_psi=[0, 0, 0]) #setup internal variables to actuate at MoveGrasper
    #SG.MoveGrasper()  # actuate soft grasper



async def HardwareInitialize():
    global SG, GC, jcSG
    SG = SoftGrasper(COM_Port='COM5', BaudRate=460800, timeout=1, controllerProfile="New") #initialize soft grasper
    GC = GantryController(comport = "COM4",homeSystem = False, initPos=[0,0,0])#, homeSystem = False,initPos=[0,0,0]  #initialize gantry controller
    jcSG = JC.Joy_SoftGrasper(SGa=SG, GantryS=GC) #initialize joystick control of soft grasper and gantry controller

async def program_loop():
    global SG, GC, jcSG, updatedSoftGrasper
    #await sio.emit('gantry position commands', 'Gantry pos')
    #await sio.emit('soft grasper commands', 'Soft Grasper Commands')
    try:
        while(True):
            [buttonVal, AxesPos] = jcSG.eventLoop()  # run event loop to determine what values and axes to execute
            jcSG.ExecuteButtonFunctions(buttonVal,
                                        AxesPos)  # execute Button functions defined by the button values and axes positions
            posInc,feedrate_mmps = jcSG.calcPositionIncrement(AxesPos[0], AxesPos[1])  # get the joystick position for x, y and z, corrected for the flip on the x axis
            GC.calculateIncrementalMove(*posInc) #will set the GC.goalPos with the appropriate increments
            GC.setXYZ_Position(*GC.goalPos,feedrate_mmps*60)  # absolute move of the gantry

            SG.MoveGrasper()  # jcRG.executeButtonFunctions should update SGa with the the pressures, this command sends the appropriate commands to the grasper over serial


            if updatedSoftGrasper == True:
                print("UPDATED Soft Grasper has happened")
                updatedSoftGrasper = False
            # Rumble feedback based on pressure change
            pressureThreshold = [0.4, 0.4,
                                 0.4]  # change in pressure threshold in psi above which to register changes in pressure
            rumbleValue = [(x - pressureThreshold[i]) / 1.5 if x >= pressureThreshold[i] else 0 for (i, x) in
                           enumerate(SG.changeInPressure)]
            jcSG.rumbleFeedback(max(rumbleValue), max(rumbleValue), 1000)



            await asyncio.sleep(0.001) #allow other tasks to run
            print('ProgramLoop')


    except KeyboardInterrupt:
        print('Exiting loop')
        return






async def start_Program():
    await HardwareInitialize()
    await sio.connect('http://localhost:5000')
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