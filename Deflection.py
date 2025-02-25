import pandas as pd
from lasso.dyna import D3plot, ArrayType, Binout, FilterType
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib.animation import FuncAnimation
import os

class Deflection:
    def __init__(self, simu, name):
        self.simu = simu
        self.name = name
        self.nodes = np.array([], dtype=int)  # Stocke les IDs des nœuds

    def get_node_mask_from_id(self, node_id_set):
        """Renvoie un masque booléen indiquant quels nœuds sont présents."""
        return np.isin(self.simu.d3plot.arrays[ArrayType.node_ids], node_id_set)
    
    def add_nodes_by_id(self, node_id_set):
        """Ajoute des nœuds au tableau self.nodes en évitant les doublons."""
        node_id_set = np.array(node_id_set, dtype=int)  # Conversion en numpy array
        self.nodes = np.unique(np.concatenate((self.nodes, node_id_set)))  # Fusion sans doublons
        print(f"✅ {len(node_id_set)} nœud(s) ajouté(s). Nouveaux IDs : {self.nodes}")

    def remove_nodes_by_id(self, node_id_set):
        node_id_set = np.array(node_id_set, dtype=int)
        self.nodes = self.nodes[~np.isin(self.nodes, node_id_set)]  # Supprime les IDs présents
        print(f"❌ {len(node_id_set)} nœud(s) supprimé(s). Nœuds restants : {self.nodes}")

    def add_nodes_by_part(self, part_id_set):
        mask_node = self.simu.d3plot.get_part_filter(FilterType.NODE, part_id_set)
        nodes_id = self.simu.d3plot.arrays[ArrayType.node_ids][mask_node]
        self.add_nodes_by_id(nodes_id)  # Réutilisation de la fonction d'ajout

    def remove_nodes_by_part(self, part_id_set):
        mask_node = self.simu.d3plot.get_part_filter(FilterType.NODE, part_id_set)
        nodes_id = self.simu.d3plot.arrays[ArrayType.node_ids][mask_node]
        self.remove_nodes_by_id(nodes_id)  # Réutilisation de la fonction de suppression

    def view(self, plane="xy", substract_CoM=False, animated=True, time_frame_index=[], save=False, saveName="nodes_deflection.gif"):
        # Get node displacements
        node_displacements = self.simu.d3plot.arrays[ArrayType.node_displacement][:, self.get_node_mask_from_id(self.nodes), :]
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

    def calculate_deflection(self, node_ref_name="T8", norm=True):
        
        nodes = self.simu.d3plot.arrays[ArrayType.node_displacement][:, self.get_node_mask_from_id(self.nodes)]

        content = Binout("Frontal_Kroell/binout")
        print(content.read('nodout', 'legend_ids'))

        if np.shape(nodes)[1]==0:
            print("No nodes to calculate deflection")
            exit()

        if node_ref_name=="T8":
            # Get T8 center of mass node evolution
            # T8 Cortical (Left 89000801, 89500801)
            # Replace by simu.parts["NUM"]
            mask_ref = self.simu.d3plot.get_part_filter(FilterType.NODE, [89000801, 89500801]) 
            nodes_ref = self.simu.d3plot.arrays[ArrayType.node_displacement][:, mask_ref]
            node_ref = np.mean(nodes_ref, axis=1)

        if node_ref_name=="T12":
            mask_ref = self.simu.d3plot.get_part_filter(FilterType.NODE, [89001201, 89501201])
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

        # Animation setup
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter([], [], c=[], cmap='viridis', s=20, alpha=0.8)
        ax.set_title("Déflexion des nœuds - Frame 0")
        ax.set_xlabel("Position X (mm)")
        ax.set_ylabel("Position Y (mm)")
        ax.axis('equal')
        ax.grid(True)
        cbar = plt.colorbar(scatter, ax=ax, label="Déflexion (mm ou unité de la simulation)")

        def update(frame):
            ax.set_title(f"Déflexion des nœuds - Frame {frame}")
            deflection = deflection_all_nodes[frame]
            nodes_positions = nodes[frame, :, [0, 1]]  # Projection sur le plan X-Z
            #scatter.set_offsets(nodes_positions)
            scatter.set_array(deflection)
            return scatter,

        ani = FuncAnimation(fig, update, frames=deflection_all_nodes.shape[0], blit=True, interval=200)
        ani.save("deflection_animation.mp4", fps=10, dpi=150)
        
        return deflection_all_nodes
     
    def get_deflection(self, norm=True, type="max", node_ref_name="T8", threshold=0.2):

        deflection_all_nodes=self.calculate_deflection(node_ref_name, norm)
        
        # Cas 1️⃣ : Déflexion maximale et ID du nœud correspondant
        if type == "max":
            max_deflection = np.max(deflection_all_nodes)
            time_step, node_idx = np.unravel_index(np.argmax(deflection_all_nodes), deflection_all_nodes.shape)
            max_node_id = self.nodes[node_idx]  # ID du nœud correspondant
            return max_node_id, max_deflection

        # Cas 2️⃣ : Liste des nœuds dépassant le seuil
        elif type == "threshold":
            mask_exceeding = np.any(deflection_all_nodes > threshold, axis=0)  # Masque booléen des nœuds dépassant le seuil
            exceeding_nodes = self.nodes[mask_exceeding]  # IDs des nœuds dépassant le seuil
            exceeding_values = np.max(deflection_all_nodes[:, mask_exceeding], axis=0)  # Valeurs max par nœud
            return exceeding_nodes, exceeding_values
            # Get index

    def calculate_VCmax(self, node_ref_name="T8"):
        """
        Calcule la vitesse de déflexion maximale (VCmax) pour chaque élément en fonction du temps.

        Returns:
            - VCmax (float): La valeur maximale du Viscous Criterion.
            - node_id_max (int): L'ID du nœud ayant le VCmax maximal.
        """
        # Récupération de la déflexion brute et normalisée
        deflection_all_nodes = self.calculate_deflection(node_ref_name, norm=False)  # (timesteps, nb_elem)
        deflection_norm_all_nodes = self.calculate_deflection(node_ref_name, norm=True)

        print(np.shape(deflection_all_nodes))
        print(np.shape(deflection_norm_all_nodes))

        # 🔹 Récupérer le pas de temps du modèle
        time_array = self.simu.d3plot.arrays[ArrayType.global_timesteps]  # (timesteps,)
        dt = np.mean(np.diff(time_array))  # Calcul du pas de temps moyen
        print(f"📌 Pas de temps du modèle : {dt:.6f} s")

        # 🔹 Calcul de la vitesse de déflexion
        velocity_deflection = np.gradient(deflection_all_nodes, dt, axis=0)  # Dérivée temporelle

        # 🔹 Calcul du Viscous Criterion
        VC_all_nodes = np.abs(velocity_deflection * deflection_norm_all_nodes)  # (timesteps, nb_elem)

        # 🔹 Trouver le nœud avec le VCmax maximum
        VCmax_values = np.max(VC_all_nodes, axis=0)  # Max de chaque élément
        node_index_max = np.argmax(VCmax_values)  # Index du max global
        node_id_max = self.nodes[node_index_max]  # ID du nœud correspondant
        VCmax = VCmax_values[node_index_max]  # Valeur max du VCmax

        print(f"📌 VCmax Global : {VCmax:.6f}")
        print(f"📌 Nœud ayant le VCmax max : {node_id_max}")

        # 🔹 Affichage du VCmax pour le nœud ayant le max
        plt.figure(figsize=(10, 5))
        plt.plot(time_array, VC_all_nodes[:, node_index_max]/1000, label=f"VC {node_id_max}", color="red")
        #plt.plot(time_array, velocity_deflection[:, node_index_max]/1000, label=f"velocity {node_id_max}", color="green")
        #plt.plot(time_array, deflection_norm_all_nodes[:, node_index_max], label=f"deflection {node_id_max}", color="blue")
        #plt.plot(time_array, deflection_all_nodes[:, node_index_max]/1000, label=f"deflection {node_id_max}", color="pink")
        plt.xlabel("Temps (s)")
        plt.ylabel("Viscous Criterion [m/s]")
        plt.title(f"Évolution du VCmax pour le nœud {node_id_max}")
        plt.legend()
        plt.grid()
        plt.show()

        return VCmax, node_id_max