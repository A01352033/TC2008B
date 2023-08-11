# Importamos las clases que se requieren para manejar los agentes (Agent) y su entorno (Model).
# Cada modelo puede contener múltiples agentes.
from mesa import Agent, Model
from mesa.model import Model 

# Con ''SimultaneousActivation, hacemos que todos los agentes se activen ''al azar''.
from mesa.time import RandomActivation

# Haremos uso de ''DataCollector'' para obtener información de cada paso de la simulación.
from mesa.datacollection import DataCollector

# matplotlib lo usaremos crear una animación de cada uno de los pasos del modelo.
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128

# Importamos los siguientes paquetes para el mejor manejo de valores numéricos.
import numpy as np
import pandas as pd

# Definimos otros paquetes que vamos a usar para medir el tiempo de ejecución de nuestro algoritmo.
import time
import datetime

# Nearest neighbors lo usaremos para obtener los vecinos más cercanos de cada agente.
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(metric='euclidean')

class FlockAgent(Agent):
    def __init__(self, id, model, x, y, width, height):
        super().__init__(id, model)

        self.position = np.array((x, y), dtype=np.float64)

        vec = (np.random.rand(2) - 0.5) * 10
        self.velocity = np.array(vec, dtype=np.float64)

        vec = (np.random.rand(2) - 0.5) / 2
        self.aceleration = np.array(vec, dtype=np.float64)

        self.max_force = 0.3

        self.max_speed = 5

        self.width = width

        self.height = height

    def step(self):
        self.check_edges()
        self.check_with_neighbors()

        self.position += self.velocity
        self.velocity += self.aceleration

        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.velocity / (np.linalg.norm(self.velocity) * self.max_speed)

        self.aceleration = np.array((0, 0), dtype=np.float64)

    def check_edges(self):
        if self.position.flatten()[0] > self.width:
            self.position[0] = 0
        elif self.position.flatten()[1] < 0:
            self.position[0] = self.width

        if self.position.flatten()[1] > self.height:
            self.position[1] = 0
        elif self.position.flatten()[1] < 0:
            self.position[1] = self.height