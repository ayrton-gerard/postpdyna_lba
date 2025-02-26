import pandas as pd
from lasso.dyna import Binout
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import os

# Charger le fichier binout
binout_file = "Frontal_Kroell_v4/binout"
content = Binout(binout_file)

print(content.read('nodout'))
# Lire les données de déplacement des nœuds
time = content.read('nodout', 'time')  # Temps associé aux données
x_coordinate = content.read('nodout', 'x_coordinate')  # Coordonnées en x
y_coordinate = content.read('nodout', 'y_coordinate')  # Coordonnées en y
z_coordinate = content.read('nodout', 'z_coordinate')  # Coordonnées en z
node_ids = content.read('nodout', 'legend_ids')  # IDs des nœuds
