from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams["animation.embed_limit"] = 2**128

import numpy as np
import pandas as pd

import time
import datetime

from sklearn.neighbors import NearestNeighbors

neighborhood = NearestNeighbors(metric="euclidean")


class FlockAgent(Agent):
    def __init__(self, id, model, x, y, width, heigth):
        super().__init__(id, model)
        self.position = np.array((x, y), dtype=np.float64)

        vec = (np.random.rand(2) - 0.5) * 10

        self.velocity = np.array(vec, dtype=np.float64)

        vec = (np.random.rand(2) - 0.5) / 2

        self.acceleration = np.array(vec, dtype=np.float64)

        self.width = width
        self.heigth = heigth

        self.max_speed = 5.0
        self.max_force = 0.3

        self.perception = 50.0

    def step(self):
        self.check_edges()
        self.check_with_neighbors()

        self.position += self.velocity
        self.velocity += self.acceleration

        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.max_speed

        self.acceleration = np.array((0, 0), dtype=np.float64)

    def check_edges(self):
        if self.position.flatten()[0] > self.width:
            self.position[0] = 0
        elif self.position.flatten()[0] < 0:
            self.position[0] = self.width

        if self.position.flatten()[1] > self.heigth:
            self.position[1] = 0
        elif self.position.flatten()[1] < 0:
            self.position[1] = self.heigth

    def check_with_neighbors(self):
        aligment = self.align()
        cohesion = self.cohesion()
        separation = self.separation()

        self.acceleration += aligment
        self.acceleration += cohesion
        self.acceleration += separation

    def align(self):
        steering = np.array((0, 0), dtype=np.float64)
        avg_vector = np.array((0, 0), dtype=np.float64)
        result = neighborhood.radius_neighbors([self.position], self.perception)[1][0]

        for idx in result:
            avg_vector += self.model.schedule.agents[idx].velocity

        total = len(result)
        if total > 0:
            avg_vector /= total # average
            avg_vector = avg_vector / np.linalg.norm(avg_vector) * self.max_speed # normalize
            steering = avg_vector - self.velocity # steering = desired - velocity

        return steering
    
    def cohesion(self):
        steering = np.array((0, 0), dtype=np.float64)
        center_of_mass = np.array((0, 0), dtype=np.float64)

        result = neighborhood.radius_neighbors([self.position], self.perception)[1][0]

        for idx in result:
            center_of_mass += self.model.schedule.agents[idx].position

        total = len(result)
        if total > 0:
            center_of_mass /= total
            vec_to_com = center_of_mass - self.position
            if np.linalg.norm(vec_to_com) > 0:
                vec_to_com = (vec_to_com / np.linalg.norm(vec_to_com)) * self.max_speed
            steering = vec_to_com - self.velocity

            if np.linalg.norm(steering) > self.max_force:
                steering = steering / np.linalg.norm(steering) * self.max_force

        return steering
    
    def separation(self):
        steering = np.array((0, 0), dtype=np.float64)
        avg_vector = np.array((0, 0), dtype=np.float64)

        result = neighborhood.radius_neighbors([self.position], self.perception)[1][0]

        for idx in result:
                diff = self.position - self.model.schedule.agents[idx].position
                avg_vector += diff

        total = len(result)
        if total > 0:
            avg_vector /= total
            if np.linalg.norm(avg_vector) > 0:
                avg_vector = (avg_vector / np.linalg.norm(avg_vector)) * self.max_speed
            steering = avg_vector - self.velocity

            if np.linalg.norm(steering) > 0:
                steering = steering / np.linalg.norm(steering) * self.max_force

        return steering

def get_position(model):
    result = []
    for agent in model.schedule.agents:
        result.append(agent.position)
        result = np.array(result)
    return result

class FlockModel(Model):
    def __init__(self, num_agents, width, heigth):
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={"Position": get_position}
        )

        data = None
        for i in range (num_agents):
            x = np.random.rand() * width
            y = np.random.rand() * heigth
            agent = FlockAgent(i, self, x, y, width, heigth)
            self.schedule.add(agent)
            
            if data is None:
                data = np.array([[ x , y ]] )
            else:
                data = np.concatenate((data, [[ x , y ]]))

        neighborhood.fit(data)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

WIDTH = 100
HEIGTH = 100
NUM_AGENTS = 100
MAX_ITERATIONS = 100

model = FlockModel(NUM_AGENTS, WIDTH, HEIGTH)
for i in range(MAX_ITERATIONS):
    model.step()

all_positions = model.datacollector.get_model_vars_dataframe()

fig, ax = plt.subplots(figsize=(5, 5))
scatter = ax.scatter(all_positions.iloc[0][0][: , 0], all_positions.iloc[0][0][: , 1], s=10, edgecolors="k")
ax.axis([0, WIDTH, 0, HEIGTH])

def animate(i):
    scatter.set_offsets(all_positions.iloc[i][0])

anim = animation.FuncAnimation(fig, animate, frames=MAX_ITERATIONS, interval=20, blit=True)

# def main():
#     num_agents = 100
#     width = 100
#     heigth = 100

#     model = FlockModel(num_agents, width, heigth)
#     for i in range(100):
#         model.step()

#     fig = plt.figure()
#     ax = plt.axes(xlim=(0, width), ylim=(0, heigth))

#     scatter = ax.scatter([], [], s=10)

#     def init():
#         scatter.set_offsets([])
#         return scatter,

#     def animate(i):
#         data = model.datacollector.get_model_vars_dataframe()
#         pos = data["Position"][i]
#         scatter.set_offsets(pos)
#         return scatter,

#     anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=20, blit=True)

#     plt.show()

# if __name__ == "_main_":
#     main()