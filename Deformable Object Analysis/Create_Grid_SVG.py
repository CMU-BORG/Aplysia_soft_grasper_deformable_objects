from DeformableClass import DeformableClass
import pyvista as pv
import copy
import pandas as pd
import numpy as np
import os
from svgutils.compose import *

if __name__ == "__main__":

    #Create the Open loop figure
    Figure("3000px", "3000px",
           SVG("Objects/Pictures/Soft Open 45.0 n1.svg"),
           SVG("Objects/Pictures/Rigid Open 45.0 n1.svg"),
           SVG("Objects/Pictures/Soft w Rigid Jaws Open 45.0 n1.svg"),
           SVG("Objects/Pictures/Soft Open 40.0 n1.svg"),
           SVG("Objects/Pictures/Rigid Open 40.0 n1.svg"),
           SVG("Objects/Pictures/Soft w Rigid Jaws Open 40.0 n1.svg"),
           SVG("Objects/Pictures/Soft Open 35.0 n1.svg"),
           SVG("Objects/Pictures/Rigid Open 35.0 n1.svg"),
           SVG("Objects/Pictures/Soft w Rigid Jaws Open 35.0 n2.svg")
           ).tile(3, 3).save("test.svg")

    #Create the Closed loop figure
    Figure("2000px", "4000px",
           SVG("Objects/Pictures/Soft Closed 15.0 n2.svg"),
           SVG("Objects/Pictures/Rigid Closed 15.0 n1.svg"),
           SVG("Objects/Pictures/Soft Closed 7.5 n1.svg"),
           SVG("Objects/Pictures/Rigid Closed 7.5 n1.svg"),
           SVG("Objects/Pictures/Soft Closed 5.0 n2.svg"),
           SVG("Objects/Pictures/Rigid Closed 5.0 n1.svg"),
           SVG("Objects/Pictures/Soft Closed 4.28 n1.svg"),
           SVG("Objects/Pictures/Rigid Closed 4.28 n1.svg"),
           ).tile(2, 4).save("test_closed.svg")