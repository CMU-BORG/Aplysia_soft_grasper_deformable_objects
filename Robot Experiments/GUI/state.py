class State:

    def __init__(self, grasper_type):
        self.gantry = [0, 0]
        if grasper_type == 'rigid':
            self.grasper_l = 0
            self.grasper_r = 0
        elif grasper_type == 'soft':
            self.damping = 0
            self.stiffness = 0
            self.power = 0

        # 0: no attempt, 1: succesful attempt, -1: failed attempt
        self.objects = [0, 0, 0,
                        0, 0, 0,
                        0, 0, 0]

        
