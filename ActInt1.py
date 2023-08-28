# Autor: Manuel Villalpando Linares
# Fecha: 27 de agosto del 2023
# Clase: TC2008B

import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

EMPTY = 0
FISH = 1
SHARK = 2

color_palette = ['#66CCCC', '#FF6666', '#66FF66']
num_colors = 3
ecosystem_cmap = ListedColormap(color_palette, N=num_colors)

NUM_TIME_STEPS = 400
SEED = 10
random.seed(SEED)

class Organism():
    def __init__(self, species, x, y, energy, reproduction_limit):
        self.species = species
        self.x, self.y = x, y
        self.energy = energy
        self.reproduction_limit = reproduction_limit
        self.reproduction_counter = 0
        self.is_dead = False

class Ecosystem():
    def __init__(self, width=75, height=50):
        self.width, self.height = width, height
        self.total_cells = width * height
        self.grid = [[EMPTY]*width for _ in range(height)]
        self.organisms = []

    def get_neighbors(self, x, y):
        neighbors = {}
        for dx, dy in ((0,-1), (1,0), (0,1), (-1,0)):
            xp, yp = (x+dx) % self.width, (y+dy) % self.height
            neighbors[xp,yp] = self.grid[yp][xp]
        return neighbors

    def evolve_organism(self, organism):
        neighbors = self.get_neighbors(organism.x, organism.y)
        organism.reproduction_counter += 1
        moved = False
        if organism.species == SHARK:
            try:
                xp, yp = random.choice([pos
                            for pos in neighbors if neighbors[pos]!=EMPTY
                                                and neighbors[pos].species==FISH])
                organism.energy += 2
                self.grid[yp][xp].is_dead = True
                self.grid[yp][xp] = EMPTY
                moved = True
            except IndexError:
                pass

        if not moved:
            try:
                xp, yp = random.choice([pos
                            for pos in neighbors if neighbors[pos]==EMPTY])
                if organism.species != FISH:
                    organism.energy -= 1
                moved = True
            except IndexError:
                xp, yp = organism.x, organism.y

        if organism.energy < 0:
            organism.is_dead = True
            self.grid[organism.y][organism.x] = EMPTY
        elif moved:
            x, y = organism.x, organism.y
            organism.x, organism.y = xp, yp
            self.grid[yp][xp] = organism
            if organism.reproduction_counter >= organism.reproduction_limit:
                organism.reproduction_counter = 0
                self.introduce_organism(organism.species, x, y)
            else:
                self.grid[y][x] = EMPTY

    def introduce_organism(self, species, x, y):
        organism = Organism(species, x, y,
                            initial_energy_levels[species],
                            reproduction_limits[species])
        self.organisms.append(organism)
        self.grid[y][x] = organism

    def get_ecosystem_image(self):
        img_array = self.get_ecosystem_array()
        fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
        ax = fig.add_subplot(111)
        ax.imshow(img_array, interpolation='nearest', cmap=ecosystem_cmap)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        return fig

    def populate_ecosystem(self, num_fish=120, num_sharks=40):
        self.num_fish, self.num_sharks = num_fish, num_sharks

        def place_organisms(num_organisms, species):
            for _ in range(num_organisms):
                while True:
                    x, y = divmod(random.randrange(self.total_cells), self.height)
                    if not self.grid[y][x]:
                        self.introduce_organism(species, x, y)
                        break

        place_organisms(self.num_fish, FISH)
        place_organisms(self.num_sharks, SHARK)

    def evolve_ecosystem(self):
        random.shuffle(self.organisms)
        num_organisms = len(self.organisms)
        for i in range(num_organisms):
            organism = self.organisms[i]
            if organism.is_dead:
                continue
            self.evolve_organism(organism)

        self.organisms = [organism for organism in self.organisms
                          if not organism.is_dead]

    def save_ecosystem(self, filename):
        fig = self.get_ecosystem_image()
        plt.savefig(filename, dpi=72, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def get_ecosystem_array(self):
        return [[self.grid[y][x].species if self.grid[y][x] else 0
                    for x in range(self.width)] for y in range(self.height)]

    def display_ecosystem(self):
        fig = self.get_ecosystem_image()
        plt.show()
        plt.close(fig)

initial_energy_levels = {FISH: 20, SHARK: 3}
reproduction_limits = {FISH: 4, SHARK: 12}

ecosystem = Ecosystem()
ecosystem.populate_ecosystem()

fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
ax = fig.add_subplot(111)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
img = ax.imshow(ecosystem.get_ecosystem_array(), interpolation='nearest', cmap=ecosystem_cmap)

def update(frame):
    ecosystem.evolve_ecosystem()
    img.set_array(ecosystem.get_ecosystem_array())
    ax.set_title(f"Step: {frame + 1}")
    return img,

animation = FuncAnimation(fig, update, frames=NUM_TIME_STEPS, repeat=False, blit=True)

plt.show()
