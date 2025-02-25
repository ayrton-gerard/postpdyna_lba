from lasso.dyna import D3plot, ArrayType, Binout, FilterType
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.cm as cm  # Pour les dégradés de couleurs
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

class Part:
    def __init__(self, simu, id=[], name="part"):
        self.id = id
        self.simu = simu
        self.name = name

        self.load_nodes()
        self.load_elements()
        self.load_elements_stress()
        self.load_elements_strain()

    def load_nodes(self):
       
        mask_node_parts = self.simu.d3plot.get_part_filter(FilterType.NODE, self.id)
        self.nodes = self.simu.d3plot.arrays[ArrayType.node_coordinates][mask_node_parts]
       
    def load_elements(self):
        mask_element_parts = self.simu.d3plot.get_part_filter(FilterType.SHELL, self.id)
        if any(mask_element_parts):
            self.elementsType="shell"
            self.elements = self.simu.d3plot.arrays[ArrayType.element_shell_ids][mask_element_parts]
        else:
            mask_element_parts = self.simu.d3plot.get_part_filter(FilterType.SOLID, self.id)
            self.elementsType="solid"
            self.elements = self.simu.d3plot.arrays[ArrayType.element_solid_ids][mask_element_parts]

    def load_elements_strain(self):     
        if self.elementsType=="shell":
            # Shape (time, nbEleme, 2 (inner surface, outer surface), 6)
            mask_element_parts = self.simu.d3plot.get_part_filter(FilterType.SHELL, self.id)
            arrayType=ArrayType.element_shell_strain
        if self.elementsType=="solid":
            mask_element_parts = self.simu.d3plot.get_part_filter(FilterType.SOLID, self.id)
            arrayType=ArrayType.element_solid_strain

        self.elements_strain = self.simu.d3plot.arrays[arrayType][:, mask_element_parts, :, :]

    def load_elements_stress(self):
        if self.elementsType=="shell":
            # Shape (time, nbEleme, 3 (mid surface, inner surface, outer surface), 6)
            mask_element_parts = self.simu.d3plot.get_part_filter(FilterType.SHELL, self.id)
            arrayType=ArrayType.element_shell_stress
        if self.elementsType=="solid":
            mask_element_parts = self.simu.d3plot.get_part_filter(FilterType.SOLID, self.id)
            arrayType=ArrayType.element_solid_stress

        self.elements_stress = self.simu.d3plot.arrays[arrayType][:, mask_element_parts, :, :]

    def set_damage(self, damage=dict()):
        # Format :: ["metric type", "threshold", "title", "units"]
        self.damage=damage

    def __str__(self):
        return "Part " + str(self.id) + " / " + self.name + " / " + self.elementsType

    def read_stress_strain_with_layer(self, quantity, layer=0):
        return quantity[:, :, layer, :]

    def get_principal_stresses(self, layer=0):
        # Layer = integration point 0, 1 or 2
        # stresses in self.stresses
        # Create a matrix of 3x3 for each timestep and each element
        elements_stress=self.read_stress_strain_with_layer(self.elements_stress, layer)

        dim1, dim2, dim3 = np.shape(elements_stress)
        stress_tensor = np.zeros((dim1, dim2, 3, 3))

        # Diagonal
        stress_tensor[:, :, 0, 0]=elements_stress[:, :, 0] # xx
        stress_tensor[:, :, 1, 1]=elements_stress[:, :, 1] # yy
        stress_tensor[:, :, 2, 2]=elements_stress[:, :, 2] # zz

        # Outer
        # Strain pour éléments Solid: εxx, εyy, εzz, εxy, εyz, εxz
        stress_tensor[:, :, 0, 1]=stress_tensor[:, :, 1, 0]=elements_stress[:, :, 3] # xy
        stress_tensor[:, :, 1, 2]=stress_tensor[:, :, 2, 1]=elements_stress[:, :, 4] # yz
        stress_tensor[:, :, 0, 2]=stress_tensor[:, :, 2, 0]=elements_stress[:, :, 5] # xz

        # Diagonalize and get the eigenvalues
        principal_stresses = np.linalg.eigvals(stress_tensor)
        principal_stresses = np.sort(principal_stresses, axis=2)

        return principal_stresses 

    def get_first_principal_stresses(self, principal_stresses):
        first_principal_stresses = np.max(principal_stresses, axis=2)
    
        # Time evolution of first principal stresses for each elem (timestep, nbElem)
        return first_principal_stresses
    
    def get_pressures(self, layer=0):
        # In MPa
        elements_stress=self.read_stress_strain_with_layer(self.elements_stress, layer)
        return np.copy(-np.mean(elements_stress[:,:,0:3], axis=2))

    def get_principal_strains(self, layer=0):

        elements_strain=self.read_stress_strain_with_layer(self.elements_strain, layer)
        # stresses in self.stresses
        # Create a matrix of 3x3 for each timestep and each element
        dim1, dim2, dim3 = np.shape(elements_strain)
        strain_tensor = np.zeros((dim1, dim2, 3, 3))

        # Diagonal
        strain_tensor[:, :, 0, 0]=elements_strain[:, :, 0] # xx
        strain_tensor[:, :, 1, 1]=elements_strain[:, :, 1] # yy
        strain_tensor[:, :, 2, 2]=elements_strain[:, :, 2] # zz

        # Outer
        # Strain pour éléments Solid: εxx, εyy, εzz, εxy, εyz, εxz
        strain_tensor[:, :, 0, 1]=strain_tensor[:, :, 1, 0]=elements_strain[:, :, 3] # xy
        strain_tensor[:, :, 1, 2]=strain_tensor[:, :, 2, 1]=elements_strain[:, :, 4] # yz
        strain_tensor[:, :, 0, 2]=strain_tensor[:, :, 2, 0]=elements_strain[:, :, 5] # xz

        # Diagonalize and get the eigenvalues
        principal_strains = np.linalg.eigvals(strain_tensor)
        principal_strains = np.sort(principal_strains, axis=2)

        return principal_strains 

    def get_first_principal_strains(self, principal_strains):
        first_principal_strains = np.max(principal_strains, axis=2)
    
        return first_principal_strains

    def plot_damage_evolution(self, quantity, quantityName=""):
        
        damage_values = quantity  # (timesteps, nb_elements)
        elements_exceeding_threshold = damage_values >= self.damage["threshold"]
        count_exceeding_elements = np.sum(elements_exceeding_threshold, axis=1)  # Somme sur les éléments (par timestep)

        # Récupérer les timesteps
        timesteps = self.simu.d3plot.arrays[ArrayType.global_timesteps]

        count_exceeding_elements=100*count_exceeding_elements/len(quantity[0, :])

        # Tracer l'évolution du nombre d'éléments
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, count_exceeding_elements, marker="o", color="red", label="Elements")
        #plt.title(f"{quantityName}, threshold of {self.damage["title"]} ({self.damage_threshold})")
        plt.xlabel("Time [s]")
        plt.ylabel("Proportion of elements >= threshold [%]")
        #plt.grid(True)
        plt.legend()
        plt.show()

    def get_effective_plastic_strain(self, layer=0):
      
        mask_shell_parts = self.simu.d3plot.get_part_filter(FilterType.SHELL, self.id)
        EPS = self.simu.d3plot.arrays[ArrayType.element_shell_effective_plastic_strain][:, mask_shell_parts]
        return EPS[:, :, layer]

    def get_von_mises_stresses(self, layer=0):  

        elements_stress=self.read_stress_strain_with_layer(self.elements_stress, layer)

        sig_xx = elements_stress[:, :, 0]
        sig_yy = elements_stress[:, :, 1]
        sig_zz = elements_stress[:, :, 2]
        sig_xy = elements_stress[:, :, 3]
        sig_yz = elements_stress[:, :, 4]
        sig_xz = elements_stress[:, :, 5]
 
        von_mises_stresses = np.sqrt(0.5*
            ((sig_xx - sig_yy) ** 2 +
            (sig_yy - sig_zz) ** 2 +
            (sig_zz - sig_xx) ** 2 +
            6 * (sig_xy**2 + sig_yz**2 + sig_xz**2))
            )
            
        return von_mises_stresses
    
    def get_von_mises_strains(self, layer=0):    

        elements_strain=self.read_stress_strain_with_layer(self.elements_strain, layer)

        # Strain pour éléments Solid: εxx, εyy, εzz, εxy, εyz, εxz
        eps_xx = elements_strain[:, :, 0]
        eps_yy = elements_strain[:, :, 1]
        eps_zz = elements_strain[:, :, 2]
        eps_xy = elements_strain[:, :, 3]
        eps_yz = elements_strain[:, :, 4]
        eps_xz = elements_strain[:, :, 5]

        # Calcul de Von Mises Strain pour solid
        von_mises_strains = np.sqrt(
            ((eps_xx - eps_yy) ** 2 +
            (eps_yy - eps_zz) ** 2 +
            (eps_zz - eps_xx) ** 2 +
            6 * ((eps_xy/2) **2 + (eps_yz/2) **2 + (eps_xz/2)**2)) / 2
            )

        return von_mises_strains

    def plot_histogram_envelope(self, quantity, quantityName="", bins=30, timestep_indices=None):
        # Utiliser tous les timesteps si aucun n'est spécifié
        if timestep_indices is None:
            timestep_indices = range(quantity.shape[0])

        timesteps = self.simu.d3plot.arrays[ArrayType.global_timesteps]
        
        # Préparer l'histogramme initial pour obtenir les bins
        all_data = quantity[timestep_indices, :].flatten()
        hist_range = (np.min(all_data), np.max(all_data))
        bin_edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
        
        plt.figure(figsize=(12, 8))
        
        # Créer une palette de couleurs basée sur le nombre de timesteps
        colors = cm.viridis(np.linspace(0, 1, len(timestep_indices)))
        
        for idx, timestep_idx in enumerate(timestep_indices):
            # Préparer les données pour le timestep courant
            timestep_quantity = quantity[timestep_idx, :].flatten()
            
            # Calculer l'histogramme
            hist_values, _ = np.histogram(timestep_quantity, bins=bin_edges)
            
            # Normaliser les valeurs de l'histogramme en pourcentage
            hist_values = hist_values / len(timestep_quantity) * 100
            
            # Calculer les positions des centres des bins
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            
            # Tracer la courbe avec une couleur différente pour chaque timestep
            plt.plot(bin_centers, hist_values, color=colors[idx], label=f"Time {timesteps[timestep_idx]:.3f} s", alpha=0.8)

        plt.title(f"Envelope of {quantityName} histograms over time")
        plt.xlabel(quantityName)
        plt.ylabel("Proportion of elements [%]")
        plt.colorbar(cm.ScalarMappable(cmap='viridis'), label='Time progression')  # Barre de couleur pour visualiser l'évolution
        plt.grid(True)
        plt.show()

    def plot_histogram_sequential(self, quantity, quantityName="", bins=30, timestep_indices=None):
        # Utiliser tous les timesteps si aucun n'est spécifié
        if timestep_indices is None:
            timestep_indices = range(quantity.shape[0])

        timesteps = self.simu.d3plot.arrays[ArrayType.global_timesteps]

        for timestep_idx in timestep_indices:
            # Préparer les données pour le timestep courant
            timestep_quantity = quantity[timestep_idx, :].flatten()

            # Créer la figure pour ce timestep
            plt.figure(figsize=(10, 6))
            plt.hist(
                timestep_quantity,
                bins=bins,
                color="blue",
                alpha=0.7,
                edgecolor="black",
                weights=np.ones_like(timestep_quantity) / len(timestep_quantity) * 100  # Convertir en pourcentage
            )
            plt.title(f"{quantityName} (Time {timesteps[timestep_idx]:.3f} s)")
            plt.xlabel("{quantityName}")
            plt.ylabel("Proportion of elements [%]")
            plt.grid(True)

            # Afficher la figure et attendre que l'utilisateur ferme avant de passer à la suivante
            plt.show()

    def create_histogram_gif(self, quantity, quantityName="", filename="von_mises_histogram.gif", bins=30, timestep_indices=None, fps=2):
        """
        Crée un GIF animé montrant l'évolution des histogrammes de Von Mises strain au cours du temps,
        avec possibilité de définir des bornes pour les axes.

        Arguments :
        - filename : Nom du fichier de sortie pour le GIF.
        - bins : Nombre de bins dans l'histogramme.
        - timestep_indices : Indices des timesteps à inclure (par défaut, tous les timesteps).
        - interval : Intervalle de temps entre les frames en millisecondes.
        - xlim : Tuple définissant les bornes de l'axe X (exemple : (0, 0.5)).
        - ylim : Tuple définissant les bornes de l'axe Y (exemple : (0, 100)).
        """
        # Utiliser tous les timesteps si aucun n'est spécifié
        if timestep_indices is None:
            timestep_indices = range(quantity.shape[0])

        timesteps = self.simu.d3plot.arrays[ArrayType.global_timesteps]
        
        # Créer la figure pour l'animation
        fig, ax = plt.subplots(figsize=(10, 6))

        def update_histogram(timestep_idx):
            """
            Met à jour l'histogramme pour le timestep donné.
            """
            ax.clear()  # Efface le contenu précédent
            timestep_quantity = quantity[timestep_idx, :].flatten()
            
            # Créer l'histogramme normalisé
            counts, bin_edges, _ = ax.hist(
                timestep_quantity,
                bins=bins,
                color="blue",
                alpha=0.7,
                edgecolor="black",
                weights=np.ones_like(timestep_quantity) / len(timestep_quantity) * 100  # Convertir en pourcentage
            )
            ax.set_title(f"{quantityName} (Time {timesteps[timestep_idx]:.3f} s)")
            ax.set_xlabel(f"{quantityName}")
            ax.set_ylabel("[%]")
            ax.grid(True)

            ax.set_xlim(np.min(quantity), np.max(quantity)+0.1*np.max(quantity))
            ax.set_ylim(0, 100)

        # Créer l'animation
        ani = animation.FuncAnimation(
            fig,
            update_histogram,
            frames=timestep_indices,
            interval=500,
            repeat=True
        )

        # Sauvegarder le GIF
        ani.save(filename, writer="imagemagick", fps=2)
        plt.close(fig)  # Fermer la figure après la sauvegarde
        print(f"GIF sauvegardé sous le nom : {filename}")

    def plot_by_nodes(self):
        # Visualisation en 3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Tracer les points (nœuds)
        ax.scatter(self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2], 
            c='b', s=10, alpha=0.8, label=self.name)

        # Ajouter des labels
        ax.set_title(self.name)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
        ax.legend()

        plt.show()
