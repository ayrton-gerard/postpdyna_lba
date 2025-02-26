import pandas as pd
from lasso.dyna import D3plot, ArrayType, Binout, FilterType
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from Part import *
from Deflection import *
from Simulation import *
from FileManager import *
from Graphics import *
from InjuryCriteria import *

fm = FileManager()
simulation = Simulation(fm.inputFile+"Simulations/Frontal_Kroell_v4/"+"d3plot")
# Create part
rib=Part(simulation, simulation.parts["left_4_rib"],  name="4th rib")

#print(np.shape(rib.elements_ids))
#print(np.shape(rib.get_von_mises_strains(layer=0)))
#rib.plot_by_nodes()

graphs=GraphicsSimpleSimu(simulation)

rib.set_metric("vm_stress", "Von-mises stress", filter="mean")
#rib.set_injury_criteria()
graphs.plot_metric_time_evolution(rib, meanstd=True)

#mask_node_parts = simulation.d3plot.get_part_filter(FilterType.NODE, simulation.parts["left_4_rib"])
#print("test1", mask_node_parts)
#print("test", simulation.d3plot.arrays[ArrayType.element_solid_node_indexes][10000, :])