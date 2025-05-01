class Gantry:

    x = 0
    y = 0

    def SetPosition(self, x, y):
        self.x = x
        self.y = y

    def GetPosition(self):
        return self.x, self.y
    
class Grasper:

    power_l = 0
    power_r = 0

    def SetPower(self, power_l, power_r):
        self.power_l = power_l
        self.power_r = power_r

    def GetPower(self):
        return self.power_l, self.power_r