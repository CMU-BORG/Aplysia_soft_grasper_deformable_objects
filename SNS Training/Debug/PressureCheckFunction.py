import numpy as np
from scipy.optimize import fsolve, root, newton,brentq, root_scalar
import cProfile
import torch
'''
Contact Force:
pressure: value in gauge psi
displacement: value in mm.  convention is 0 is the surface of the soft jaw, +ve values indicate compression into the jaw, -ve values indicate distances above the jaw
'''
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

    def ContactStiffnessCalc(self,pressure,displacement,coefficients=None):
        b = self.coefficients if coefficients is None else coefficients
        StiffVal = (b[0] + (pressure**b[4]) *b[5])*(b[2]**pressure + 2*(b[1]**(2*pressure)) * displacement)
        return StiffVal


CFcalc =ContactForce()
zeroPos = CFcalc.ContactForce_ZeroPos(pressure=10)
print(zeroPos)
fval = CFcalc.ContactForceCalc(10,5,zeroPos = zeroPos)
print(fval)

stiffval= CFcalc.ContactStiffnessCalc(2,-5)
print(stiffval)

#cProfile.run('CFcalc.ContactForceCalc(2.5,10)')
