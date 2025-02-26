import pandas as pd
from lasso.dyna import D3plot, ArrayType, Binout, FilterType
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from Part import *
from Deflection import *

class Simulation:
    def __init__(self, d3plotPath):
        # Infos extraction
        self.d3plot=D3plot(d3plotPath)
    
        # File loading
        self.units=self.read_units_file("Inputs/units.txt")
        self.expand_units()

        self.parts=self.read_parts_file("Inputs/parts.txt")
        self.critical_nodes=self.read_critical_nodes_file("Inputs/critical_nodes.txt")
        self.node_set=self.read_node_set_file("Inputs/node_set.txt")

    def expand_units(self):
        self.units["energy"]=self.units["force"]+'.'+self.units["distance"]
    
    def read_units_file(self, file_path):
        data = {}
    
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    key, value = parts
                    data[key] = value
        
        return data

    def read_parts_file(self, file_path):
        data = {}
    
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    key, value = parts
                    data[key] = int(value)
        
        return data
    
    def read_critical_nodes_file(self, file_path):
        data = {}
    
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    key, value = parts
                    data[key] = int(value)
        
        return data
    
    def read_node_set_file(self, file_path):
        nodes_dict = {}
        current_key = None

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                if "=" in line:
                    current_key = line.split("=")[0].strip()
                    nodes_dict[current_key] = []
                else:
                    numbers = [int(num) for num in line.replace(",", " ").split() if num.isdigit()]
                    nodes_dict[current_key].extend(numbers)

        return nodes_dict

# Calculate center of mass of # T8 Cortical
#mask_element_parts = sim.d3plot.get_part_filter(FilterType.NODE, [89000801, 89500801])
#print(np.shape(sim.d3plot.arrays[ArrayType.node_displacement][:, mask_element_parts]))
#ref_node = sim.calculate_CoM(sim.d3plot.arrays[ArrayType.node_coordinates][mask_element_parts])
#print(np.shape(ref_node))
# Calculate center of mass of # T8 Cortical (Left 89000801, 89500801)
# Skeletal (rib + sternum) [89004401, 89504401, 89003701, 89503701]

#print(sim.chest_bands['name'])

#deflection=Deflection(sim, "4th rib band")

#deflection.add_nodes_by_part([sim.parts["left_3_rib"], sim.parts["right_3_rib"],
#                              sim.parts["left_4_rib"], sim.parts["right_4_rib"],
#                              sim.parts["left_7_rib"], sim.parts["right_7_rib"],
#                              sim.parts["left_8_rib"], sim.parts["right_8_rib"]])
#deflection.add_nodes_by_id([sim.critical_nodes["mid_sternum"]]) # Garder qu'un seul point au niveau du sternum (pas l'appendice xyphoide) sinon trop de deflection
#deflection.add_nodes_by_id(sim.chest_bands["nodes_set_transversal"])
#print(deflection.get_deflection(norm=True, type="max", node_ref_name="T8")) # Choose between T8 or T12
#deflection.view(plane="xy", substract_CoM=False, animated=True, save=True, saveName="Results/test.gif")
#sim.chest_bands[0].view(plane='xy', save=True, animated=False, time_frame_index=[0, 1, 2])
#print(deflection.calculate_VCmax())
#print(np.max(deflection.calculate_deflection(norm=False)))

# Voir https://www.dynasupport.com/tutorial/ls-dyna-users-guide/ls-post-binary-database
# et https://github.com/open-lasso-python/lasso-python/blob/develop/lasso/dyna/array_type.pyd

# Make the part stuff
#left_lung=Part(sim, sim.parts["left_lung"], "Left lung")
#right_lung=Part(sim, sim.parts["right_lung"], "Left lung")

"""
rib=Part(sim, sim.parts["left_4_rib"], "3th rib left")
#deflection.get_deflection()
rib.create_histogram_gif(rib.get_effective_plastic_strain(), "Von mises", filename="Results/AAAA.gif")
#rib.set_damage({"metric": 'von_mises_stress', "threshold": 120, "title": 'Stress Von-Mises', "units": 'MPa'}) 
#rib.plot_damage_evolution(rib.get_von_mises_stresses(), "Von mises")

print("test", np.max(rib.get_von_mises_stresses(2)))
#left_lung.create_histogram_gif(left_lung.get_first_principal_strains(left_lung.get_principal_strains()), "P1 strain", filename="Left-lung.gif")

#right_lung.create_histogram_gif(right_lung.get_first_principal_strains(right_lung.get_principal_strains()), "P1 strain", filename="Right-lung.gif")
#heart=Part(sim, sim.parts["heart"], "Heart")
#heart.plot_histogram_envelope(heart.get_first_principal_strains(heart.get_principal_strains()), "P1 strain")
#heart.create_histogram_gif(heart.get_first_principal_strains(heart.get_principal_strains()), "P1 strain", filename="HEARTT.gif")


#rib.plot_histogram_envelope(rib.get_von_mises_stresses(), "VM Stress")
#rib.create_histogram_gif(rib.get_von_mises_stresses(), "VM Stress", filename="von-mises-rib4thleft.gif")
"""
#heart.set_damage({"metric": 'von_mises_strain', "threshold": 140, "title": 'Strain Von-Mises', "units": '-'}) # EPS (3 layers for shell, 1 for solid)
#heart.create_histogram_gif(heart.get_first_principal_strains(heart.get_principal_strains(0)), "P1 major", filename="Results/P1Heart.gif")
# Otherwise it is 3 layers for stress shell and 2 layers for strain shell
# And 1 layer for solid stress and strain

#sternum=Part(sim, [sim.parts["left_sternum_cort"], sim.parts["right_sternum_cort"]], "sternum")
#sternum.create_histogram_gif(sternum.get_von_mises_stresses(0), "EPS", filename="Results/EPSSternum.gif")

# T12 Cortical (Left 89001201, Right 89501201)
# T12 Spon (Left 89001200, Right 89501200)

# T8 Cortical (Left 89000801, Right 89500801)
# T8 Spon (Left 89000800, Right 89500800)

# 4th rib Cort (Left 89004401, Right 89504401)
# 4th rib Spon (Left 89004400, Right 89504400)

# 8th rib Cort (Left 89004801, Right 89504801)
# 8th rib Spon (Left 89004800, Right 89504800)

# Rib cortical
# Left 89004101 à 89005201 (de 100 en 100, 4101, 4201, 4601, 4901, 5001, 5101, 5201)
# Right pareil mais avec début 8950 au lieu de 8900


