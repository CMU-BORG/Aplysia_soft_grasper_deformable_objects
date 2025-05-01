from GUI.EmbeddedSystems.SNS.controller_modulation.SNS_layer import perceptor, controller_modulation, controller_closed_loop_modulation, pick_and_place, sensory_layer_2_feedback_modulation, SNS_Control_closed_loop_v1, SNS_Control_closed_loop_v2, SNS_Control_closed_loop_modulation, controller_closed_loop_v2, R, perceptor_closed_loop_modulation


OBJECT_POSITION = [0, 0, -0.315]
TARGET_POSITION = [0.15, 0.15, -0.315]
MASS = 1
SIZE_SCALING = 0.6

END_TIME = 13

OPEN_LOOP_PERCEPTOR = perceptor
CLOSED_LOOP_PERCEPTOR = perceptor_closed_loop_modulation
OPEN_LOOP_CONTROLLER = controller_modulation
CLOSED_LOOP_CONTROLLER = controller_closed_loop_v2
FORCE_THRESHOLD_GAIN = 0.1
PRESSURE_VALUE = 2.5
GRASPER_CLOSING_SPEED = 0.3
PERCEPTOR_TAU = 0.05
FORCE_MODULATION_GAIN = 5

CLOSED_LOOP_PERCEPTOR.set_modulation_gain(FORCE_MODULATION_GAIN)
CLOSED_LOOP_PERCEPTOR.set_tau(PERCEPTOR_TAU)
CLOSED_LOOP_PERCEPTOR.reset()
CLOSED_LOOP_CONTROLLER.reset()
neuron_history_closed_loop_modulation, sensory_history_closed_loop_modulation, command_history_closed_loop_modulation = pick_and_place(position_o=OBJECT_POSITION, position_t=TARGET_POSITION, perceptor=CLOSED_LOOP_PERCEPTOR, controller=CLOSED_LOOP_CONTROLLER, end_time=END_TIME, force_threshold_gain=FORCE_THRESHOLD_GAIN, grasper_closing_speed=GRASPER_CLOSING_SPEED, zero_time_constant=False, mass=MASS, sizeScaling=SIZE_SCALING)