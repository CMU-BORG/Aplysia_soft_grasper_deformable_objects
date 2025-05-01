import meshpy.tet as mp

import numpy as np
import pyvista as pv
from pyvista import examples
import tetgen
cow_mesh = examples.download_cow().triangulate()

cpos = [(13.0, 7.6, -13.85), (0.44, -0.4, -0.37), (-0.28, 0.9, 0.3)]

cpos = [
    (15.87144235049248, 4.879216382405231, -12.14248864876951),
    (1.1623113035352375, -0.7609060338348953, 0.3192320579894903),
    (-0.19477922834083672, 0.9593375398915212, 0.20428542963665386),
]

cow_mesh.plot(cpos=cpos)