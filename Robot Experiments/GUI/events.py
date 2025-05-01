from flask import request
from flask_socketio import emit, join_room
import csv

from .extensions import socketio

users = {}
curr_test = None;

def map(val, ilo, ihi, flo, fhi):
    return flo + ((fhi - flo) / (ihi - ilo)) * (val - ilo)

def mouse_to_mm(mouse_position):
    if mouse_position == None: return 0

    mouse_position = float(mouse_position)
    return int(map(mouse_position, 0, 860, -255, 255))

def mm_to_mouse(mm_position):
    if mm_position == None: return 0

    mm_position = float(mm_position)
    return int(map(mm_position, -255, 255, 0, 860))

@socketio.on("proc-join")
def handle_proc_join():
    # if 'proctor' in users:
    #     return
    
    print("proctor joined")
    join_room('proctor')
    page = 'landing'

@socketio.on("part-join")
def handle_part_join():
    # if 'participant' in users:
    #     return
    
    print("participant joined")
    join_room('participant')

@socketio.on("info-done")
def handle_info_done(user_data, trial_data):

    id_code = user_data['id_code']
    age = user_data['age']
    gender = user_data['gender']
    types = user_data['item_types']

    test = trial_data['test']
    global curr_test
    curr_test = test
    num_attempts = trial_data['num_attempts']

    socketio.emit('TestInfo', (user_data,trial_data))

    with open(f'./datalogs/{id_code}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(["id code", "age", "gender"])
        writer.writerow([id_code, age, gender])
        writer.writerow([])
        writer.writerow([f'type{i}' for i in range(len(types))])
        writer.writerow(types)

    emit("goto-action", test, room='participant')
    emit("new-target-item", test, room='proctor')


@socketio.on("sendto-action")
def handle_sendto_action(test):

    global curr_test
    curr_test = test
    emit('goto-action', test, room='participant')
    emit('new-target-item', test)


@socketio.on("gantry-move")
def handle_gantry_move(vals):

    x = -mouse_to_mm(vals['x']) #need to check if -ve is correct
    y = mouse_to_mm(vals['y'])
    print(x, ',', y)

    socketio.emit('gantry position commands', {'x': x, 'y': y})

    # GCa.getPosition()
    # curr_z = GCa.PositionArray["z"][0]
    # GCa.setXYZ_Position(x, y, curr_z) #absolute move


@socketio.on("set-contact-force-soft_hardware")
def handle_setContactForceSoft(data):
    emit("set-contact-force-soft",data,room='proctor')
    print('Emitted %f'%(data))

@socketio.on('set-gantry-marker_hardware')
def handle_getGantryMarker(data):

    global curr_test
    data['test'] = curr_test
    data['target'] = 'proc'
    emit("set-gantry-marker", data, room='participant')

    data['target'] = 'part'
    emit("set-gantry-marker", data, room='proctor')
    print('Gantry position update sent %f %f'%(data['x'],data['y']))


@socketio.on("damping-change")
def handle_damping_change(val):
    print('damping: ' + str(val))

@socketio.on("stiffness-change")
def handle_stiffness_change(val):
    print('stiffness: ' + str(val))

@socketio.on("power-change")
def handle_power_change(val):
    print('power: ' + str(val))
    socketio.emit('soft grasper commands', {'grasper_l': float(val)})
    #SGa.AbsoluteMove(closureIncrement_mm = float(val)*20/100, jawIncrement_psi = [0,0,0])
    #SGa.MoveGrasper()

@socketio.on("item-event")
def handle_item_event_proctor(data):

    n = data['n']
    typ = data['typ']
    attempts = data['att']
    test = data['test']

    emit(f"update-events-proctor-{test}", {'n':n, 'typ': typ, 'att':attempts}, room="proctor")
    emit(f"update-events-participant-{test}", {'n':n, 'typ': typ, 'att':attempts}, room="participant")
    print(f'item {n} is {typ}')

    socketio.emit('item-event_emit',data)

    attempted_non_target = False;
    for attempt in attempts:
        if attempt['n'] == n and attempt['typ'] != 'inprogress': 
            attempted_non_target = True;

    if (typ != 'inprogress' and not attempted_non_target):
        emit(f'new-target-item', test)

@socketio.on("send-target-item")
def handle_new_target_item(n, test):
    emit("update-target-item", {'n': n, 'test': test}, room="participant")

@socketio.on("pause")
def handle_pause():
    emit("pause-trial", curr_test, room="participant")

@socketio.on("continue")
def handle_continue():
    emit("continue-trial", curr_test, room="participant")

@socketio.on("cancel")
def handle_cancel():
    emit("cancel-trial", curr_test, room="participant")


# For hardware and grasper communication
@socketio.on('gantry position commands')
def sendGantryPosition(string):
    print('gantry')


@socketio.on('soft grasper commands')
def sendGantryPosition(string):
    print('softGrasper')



@socketio.on('my event')
def handle_my_custom_event(json):
    print('received json: ' + str(json))
    fsio.emit('SendInfo', 'Received')

@socketio.on('start event')
def handle_start_event(string):
    print('received string: ' + str(string))
    SID = request.sid
    socket_NAMESPACE = request.namespace
