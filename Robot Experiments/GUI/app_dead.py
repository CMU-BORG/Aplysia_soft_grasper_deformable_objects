from flask import Flask, render_template, request, url_for, flash, redirect
from datetime import datetime
#from gantry import *
# import EmbeddedSystems.Gantry.GantryController as GC
# import EmbeddedSystems.RigidGrasper.RigidGrasper as RG
# import EmbeddedSystems.SoftGrasper.SoftGrasper as SG
from state import State
import csv

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ceff418fb561ebf2572221b1f28789a36e4e30f7da4df0a8'

# Gantry / Grasper setup
#SGa = SG.SoftGrasper(COM_Port='COM5', BaudRate=115200, controllerProfile="Legacy") #soft grasper initialization
# GCa = GC.Gantry(comport = "COM4", homeSystem = True) #homeSystem = False, initPos=[0,0,0] )
# RGa = RG.RigidGrasper(BAUDRATE = 57600, DEVICEPORT = "COM6", GoalPosition1=[1500,2000], GoalPosition2 = [2120,1620])

state = State('rigid')

# date / time
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def get_date_time():
    now = datetime.now()
    date = f'{MONTHS[now.month - 1]} - {now.day} - {now.year}'

    hour = now.hour if now.hour <= 12 else now.hour - 12
    minute = now.minute if now.minute >= 10 else '0'+str(now.minute)
    ampm = 'A.M' if now.hour <= 12 else 'P.M'
    time = f'{hour}:{minute} {ampm}'

    return date, time

def log_participant_data(id_code, age, gender, types):
    with open(f'{id_code}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(["id code", "age", "gender"])
        writer.writerow([id_code, age, gender])
        writer.writerow([])
        writer.writerow([f'type{i}' for i in range(len(types))])
        writer.writerow(types)

def log_trial_data(timestamp, gantry_x, gantry_y, objects):
    pass

def map(val, ilo, ihi, flo, fhi):
    return flo + ((fhi - flo) / (ihi - ilo)) * (val - ilo)

def mouse_to_mm(mouse_position):
    mouse_position = float(mouse_position)
    return int(map(mouse_position, 0, 860, -255, 255))


def mm_to_mouse(mm):
    mm = float(mm)
    return int(map(mm, 0, 510, 0, 860));

@app.route("/", methods=('GET', 'POST'))
@app.route("/home", methods=('GET', 'POST'))
def index():
    date, time = get_date_time()

    if request.method == 'POST':
        types = []
        for i in range(9):
            types[i] = request.form[f'type{i}']
        
        id_code = request.form['id-code']
        age = request.form['age']
        gender = request.form['gender']
        
        if not id_code:
            flash('Id Code is required!')
        elif not age:
            flash('Age is required!')
        elif not gender:
            flash('Gender is required!')
        else:
            log_participant_data(id_code, age, gender, types)
            return redirect(url_for('soft_grasper'))

    return render_template("index.jinja", date=date, time=time)

@app.route("/participant/dual-grasper", methods=('GET', 'POST'))
def dual_grasper_participant():
    date, time = get_date_time()

    # read user inputs from GUI
    gantry_data = request.form.get('gantry')
    grasper_l = request.form.get('grasper_l')
    grasper_r = request.form.get('grasper_r')
    grasper_on = request.form.get('grasper_on')

    if gantry_data:
        gantry_data = gantry_data.split(',')

        state.gantry = gantry_data

        gantry_data[0] = -mouse_to_mm(gantry_data[0])
        gantry_data[1] = mouse_to_mm(gantry_data[1])

        # GCa.getPosition()
        # curr_z = GCa.PositionArray["z"][0]
        # GCa.setXYZ_Position(gantry_data[0], gantry_data[1], curr_z) #absolute move
        # GCa.incrementalMove(20,20,0,GCa.MoveSpeed/60) # relative move
    if grasper_l:
        grasper_l = round(float(grasper_l), 2)
        state.grasper_l = grasper_l
        #SGa.IncrementalMove(closureIncrement_mm = grasper_l*20/100, jawIncrement_psi = [0,0,0])
        # RG_GoalPosition = int((RGa.GoalPosition_Limits["1"][1]-RGa.GoalPosition_Limits["1"][0])*grasper_l/100 + RGa.GoalPosition_Limits["1"][0])
        # print(RG_GoalPosition)
        # RGa.SetGoalPosition(goal_position1=RG_GoalPosition,goal_position2=None)
    else:
        grasper_l = 'no input received';
    
    print('left: ' + str(grasper_l))

    if grasper_r:
        grasper_r = round(float(grasper_r), 2)
        state.grasper_r = grasper_r
        # RGa.SetGoalPosition(goal_position1=None,goal_position2=grasper_r_data)
        # print('right: ' + str(grasper_r_data))
    if grasper_on:
        print(grasper_on)
        #SGa.isActive = not SGa.isActive

        # RGa.setGoalPosition()

    # if SGa.isActive:
    #     SGa.MoveGrasper() #program is expecting 4 bytes every loop so need to continually send move command
    

    return render_template( "dual-grasper-PARTICIPANT.jinja", 
                            # mouseX=mm_to_mouse(data[0]), mouseY=mm_to_mouse(data[1]),
                            grasper_l=state.grasper_l, grasper_r=state.grasper_r,
                            date=date, time=time
                            )

#if __name__ == '__main__':
#app.run(host='0.0.0.0', port = '5000', debug=True)