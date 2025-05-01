from collections import namedtuple

Point = namedtuple("Point", "x y z")
Velocity = namedtuple("Velocity", "x y z")
Acceleration = namedtuple("Acceleration", "x y z")
GrasperContactForce = namedtuple("GrasperContactForce" , "Jaw1 Jaw2 Jaw3")