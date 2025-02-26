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

class GraphicsSimpleSimu:
    def __init__(self, simulation):
        self.simu=simulation
        pass

    def plot_metric_time_evolution(self, part, meanstd=False):
        timesteps = self.simu.d3plot.arrays[ArrayType.global_timesteps]
        plt.figure(figsize=(10, 6))
        print(np.shape(part.metric_quantity))

        if meanstd==False:
            plt.plot(timesteps, part.metric_quantity, marker="o", color="red")
        else:
            mean_values = np.mean(part.metric_quantity, axis=1)
            std_values = np.std(part.metric_quantity, axis=1)

            plt.plot(timesteps, mean_values, marker="o", color="blue", label="Mean") 
            plt.fill_between(timesteps, mean_values - std_values, mean_values + std_values, 
                         color="blue", alpha=0.2, label="Std Dev")

        plt.xlabel("Time [s]")
        plt.ylabel(part.metric_name)
        plt.grid(True)
        plt.legend()
        plt.show()

    """
    # Plot damage 3d to do
    def plot_3D_damage_node_location(self, part):
        print()


    def plot_2D_damage_evolution(self, part):
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
        # Utiliser tous les timesteps si aucun n'est spécifié
        if timestep_indices is None:
            timestep_indices = range(quantity.shape[0])

        timesteps = self.simu.d3plot.arrays[ArrayType.global_timesteps]
        
        # Créer la figure pour l'animation
        fig, ax = plt.subplots(figsize=(10, 6))

        def update_histogram(timestep_idx):
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
    """