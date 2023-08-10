from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["animation.embed_limit"] = 2**128

import numpy as np
import pandas as pd

import time
import datetime


class GameOfLifeAgent(Agent):
    def __init__(self, id, model):
        super().__init__(id, model)
        self.live = self.random.choice([0, 1])
        self.nextState = None

    def step(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )

        count = 0
        for neighbor in neighbors:
            count += neighbor.live

        self.nextState = self.live

        if self.live == 1:
            if (count < 2) or (count > 3):
                self.nextState = 0
        else:
            if count == 3:
                self.nextState = 1

    def advance(self):
        self.live = self.nextState


def getGrid(model):
    grid = np.zeros((model.grid.width, model.grid.height))
    for (agent, (x, y)) in model.grid.coord_iter():
        grid[x][y] = agent.live
    return grid

class GameOfLifeModel(Model):
    def __init__(self, width, height):
        self.grid = SingleGrid(width, height, torus=True)
        self.schedule = SimultaneousActivation(self)

        for (contents, (x, y)) in self.grid.coord_iter():
            agent = GameOfLifeAgent((x, y), self)
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)

        self.datacollector = DataCollector(
            model_reporters={"Grid": getGrid}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

GRID_SIZE = 20
MAX_GENERATIONS = 100

model = GameOfLifeModel(GRID_SIZE, GRID_SIZE)

for i in range(MAX_GENERATIONS):
    model.step()

all_grid = model.datacollector.get_model_vars_dataframe()

fig, axis = plt.subplots(figsize=(5, 5))
axis.set_xticks([])
axis.set_yticks([])
patch = plt.imshow(all_grid.iloc[0][0], cmap=plt.cm.binary)

def animate(i):
    patch.set_data(all_grid.iloc[i][0])
    return patch

anim = animation.FuncAnimation(fig, animate, frames=MAX_GENERATIONS)


# anim.save('gameOfLife.gif', writer='imagemagick', fps=10)
plt.show()

