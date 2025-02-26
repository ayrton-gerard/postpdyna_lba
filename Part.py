from lasso.dyna import D3plot, ArrayType, Binout, FilterType
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.cm as cm  # Pour les dégradés de couleurs
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

class Part:
    def __init__(self, simu, id=[], name="part",  elemID_set=[]):
        self.id = id
        self.simu = simu
        self.name = name

        self.load_nodes()
        self.load_elements_ids()
        self.load_elements_stress()
        self.load_elements_strain()

        if len(elemID_set)!=0:
            # Custom functions for solid elem
            mask_element_parts = np.isin(simu.d3plot.arrays[ArrayType.element_solid_ids], elemID_set)
            self.stresses=simu.d3plot.arrays[ArrayType.element_solid_stress][: mask_element_parts, :, :]
            self.strains=simu.d3plot.arrays[ArrayType.element_solid_strain][: mask_element_parts, :, :]

    def load_nodes(self):
        mask_node_parts = self.simu.d3plot.get_part_filter(FilterType.NODE, self.id)
        self.nodes = self.simu.d3plot.arrays[ArrayType.node_coordinates][mask_node_parts]
       
    def load_elements_ids(self):
        mask_element_parts = self.simu.d3plot.get_part_filter(FilterType.SHELL, self.id)
        if any(mask_element_parts):
            self.elementsType="shell"
            self.elements_ids = self.simu.d3plot.arrays[ArrayType.element_shell_ids][mask_element_parts]
        else:
            mask_element_parts = self.simu.d3plot.get_part_filter(FilterType.SOLID, self.id)
            self.elementsType="solid"
            self.elements_ids = self.simu.d3plot.arrays[ArrayType.element_solid_ids][mask_element_parts]

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

    def set_metric(self, metric_code, metric_name, filter):
        match metric_code:
            case "vm_stress":
                quantity = np.stack([
                                    self.get_von_mises_stresses(0),
                                    self.get_von_mises_stresses(1),
                                    self.get_von_mises_stresses(2)
                                ], axis=2) 
                self.metric_unit=self.simu.units["sigma"]
               
            case "vm_strain":
                quantity = np.stack([
                                    self.get_von_mises_strains(0),
                                    self.get_von_mises_strains(1),
                                ], axis=2) 
                self.metric_unit="[-]"
                
            case "pressure":
                quantity = np.stack([
                                    self.get_pressures(0),
                                    self.get_pressures(1),
                                    self.get_pressures(2)
                                ], axis=2)
                self.metric_unit=self.simu.units["sigma"]
                
            case "EPS":
                quantity = np.stack([
                                    self.get_effective_plastic_strain(0),
                                    self.get_effective_plastic_strain(1),
                                    self.get_effective_plastic_strain(2)
                                ], axis=2) 
                self.metric_unit="[-]"
                
            case "P1_stress":
                quantity = np.stack([
                                    self.get_first_principal_stresses(self.get_principal_stresses(0)),
                                    self.get_first_principal_stresses(self.get_principal_stresses(1)),
                                    self.get_first_principal_stresses(self.get_principal_stresses(2))
                                ], axis=2) 
                self.metric_unit=self.simu.units["sigma"]
               
            case "P1_strain":
                quantity = np.stack([
                                    self.get_first_principal_strains(self.get_principal_strains(0)),
                                    self.get_first_principal_strains(self.get_principal_strains(1)),
                                ], axis=2)
                self.metric_unit="[-]" 
                
                

        match filter:
            case "mean":
                self.metric_quantity=np.mean(quantity, axis=2)
            case "max":
                self.metric_quantity=np.max(quantity, axis=2)
            case "min":
                self.metric_quantity=np.min(quantity, axis=2)

        self.metric_name=metric_name

    def set_injury_criteria(self, injury_criteria):
        self.injury_criteria=injury_criteria