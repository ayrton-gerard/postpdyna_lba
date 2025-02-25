import pandas as pd
from lasso.dyna import D3plot, ArrayType, Binout, FilterType
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

class ChestBand:
    def __init__(self, simu, name, node_set):
        self.simu = simu
        self.name = name
        self.node_set = node_set

    def get_node_mask(self, node_set):
        return np.isin(self.simu.d3plot.arrays[ArrayType.node_ids], node_set)

    def view(self, plane="xy", substract_CoM=False, animated=True, time_frame_index=[], save=False, saveName="chest_band.gif"):

        # Get node displacements
        node_displacements = self.simu.d3plot.arrays[ArrayType.node_displacement][:, self.get_node_mask(self.node_set), :]
        CoM = np.mean(node_displacements, axis=1)
    
        if substract_CoM:
            node_displacements=node_displacements-CoM[:, np.newaxis, :]

        # Get frame size
        min_x = np.min(node_displacements[:, :, 0])
        max_x = np.max(node_displacements[:, :, 0])
        
        min_y = np.min(node_displacements[:, :, 1])
        max_y = np.max(node_displacements[:, :, 1])
        
        min_z = np.min(node_displacements[:, :, 2])
        max_z = np.max(node_displacements[:, :, 2])

        match plane:
            case 'xy':
                abs = node_displacements[:, :, 0]
                ord = node_displacements[:, :, 1]
                abs_lim = [min_x, max_x]
                ord_lim = [min_y, max_y]

            case 'xz':
                abs = node_displacements[:, :, 0]
                ord = node_displacements[:, :, 2]
                abs_lim = [min_x, max_x]
                ord_lim = [min_z, max_z]

        if animated:
            fig, ax = plt.subplots()
            ax.set_xlim(abs_lim)  # Ajuste selon ton cas
            ax.set_ylim(ord_lim)
            scat = ax.scatter([], [])

            def update(frame):
                ax.clear()
                ax.set_xlim(abs_lim)  # Ajuste selon ton cas
                ax.set_ylim(ord_lim)
                ax.set_title(f"Frame {frame}")
                ax.scatter(abs[frame, :], ord[frame, :], color='blue')

            ani = animation.FuncAnimation(fig, update, frames=node_displacements.shape[0], repeat=True)

            if save:
                ani.save(saveName, writer="pillow", fps=10)
                print("Animation saved : ", saveName)
        else:
            # Simple plot at time
            for t in time_frame_index:
                plt.figure()
                plt.scatter(abs[t, :], ord[t, :])
                plt.xlim(abs_lim)  # Ajuste selon ton cas
                plt.ylim(ord_lim)
                plt.show()

    def get_max_deflection(self, norm=True, skeletal=False, node_ref="T8"):

        # If norm==True, result in %
        if skeletal==True:
            # 4th rib left/right and sternum
            mask_node = self.simu.d3plot.get_part_filter(FilterType.NODE, [89004401, 89504401, 89003701, 89503701])
            nodes = self.simu.d3plot.arrays[ArrayType.node_displacement][:, mask_node]
        else:
            nodes = self.simu.d3plot.arrays[ArrayType.node_displacement][:, self.get_node_mask(self.node_set)]

        if node_ref=="T8":
            # Get T8 center of mass node evolution
            # T8 Cortical (Left 89000801, 89500801)
            mask_ref = self.simu.d3plot.get_part_filter(FilterType.NODE, [89000801, 89500801])
            
            nodes_ref = self.simu.d3plot.arrays[ArrayType.node_displacement][:, mask_ref]
            node_ref = np.mean(nodes_ref, axis=1)
         

        # Get initial state
        init_distances=np.linalg.norm(nodes[0,:,:]-node_ref[0, np.newaxis,:], axis=1) # Shape (nbelem, 1)

        # Calculate deflection 
        deflection_vector_all_nodes = nodes-node_ref[:, np.newaxis, :] # Shape (timesteps, nbelem, 3)

        # Calcul de la norme de la déflexion pour chaque élément à chaque pas de temps
        deflection_all_nodes=(init_distances[np.newaxis,:]-np.linalg.norm(deflection_vector_all_nodes, axis=2))

        if norm:
            deflection_all_nodes = (init_distances[np.newaxis,:]-np.linalg.norm(deflection_vector_all_nodes, axis=2))/init_distances[np.newaxis, :] # Shape (27, nbelem, 1)

        # Max deflection through time and elements
        max_deflection = np.max(deflection_all_nodes)

        return max_deflection