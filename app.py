import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

EMPTY = 0
FISH = 1
SHARK = 2

colors = ['#FFFFFF', '#ff69b4', '#ffd700']
n_bin = 3
cm = LinearSegmentedColormap.from_list(
        'wator_cmap', colors, N=n_bin)

MAX_CHRONONS = 400
SAVE_EVERY = 1
SEED = 10
random.seed(SEED)

initial_energies = {FISH: 20, SHARK: 3}
fertility_thresholds = {FISH: 4, SHARK: 12}

class Creature():
    def __init__(self, id, x, y, init_energy, fertility_threshold):
        self.id = id
        self.x, self.y = x, y
        self.energy = init_energy
        self.fertility_threshold = fertility_threshold
        self.fertility = 0
        self.dead = False


class World():
    def __init__(self, width=75, height=50):
        self.width, self.height = width, height
        self.ncells = width * height
        self.grid = [[EMPTY]*width for y in range(height)]
        self.creatures = []

    def spawn_creature(self, creature_id, x, y):
        creature = Creature(creature_id, x, y,
                            initial_energies[creature_id],
                            fertility_thresholds[creature_id])
        self.creatures.append(creature)
        self.grid[y][x] = creature

    def populate_world(self, nfish=120, nsharks=40):
        self.nfish, self.nsharks = nfish, nsharks

        def place_creatures(ncreatures, creature_id):
            for i in range(ncreatures):
                while True:
                    x, y = divmod(random.randrange(self.ncells), self.height)
                    if not self.grid[y][x]:
                        self.spawn_creature(creature_id, x, y)
                        break

        place_creatures(self.nfish, FISH)
        place_creatures(self.nsharks, SHARK)

    def get_world_image_array(self):
        return [[self.grid[y][x].id if self.grid[y][x] else 0
                    for x in range(self.width)] for y in range(self.height)]

    def get_world_image(self):
        im =  self.get_world_image_array()
        fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
        ax = fig.add_subplot(111)
        ax.imshow(im, interpolation='nearest', cmap=cm)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        return fig

    def show_world(self):
        fig = self.get_world_image()
        plt.show()
        plt.close(fig)

    def save_world(self, filename):
        fig = self.get_world_image()
        plt.savefig(filename, dpi=72, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def get_neighbours(self, x, y):
        neighbours = {}
        for dx, dy in ((0,-1), (1,0), (0,1), (-1,0)):
            xp, yp = (x+dx) % self.width, (y+dy) % self.height
            neighbours[xp,yp] = self.grid[yp][xp]
        return neighbours

    def evolve_creature(self, creature):
        neighbours = self.get_neighbours(creature.x, creature.y)
        creature.fertility += 1
        moved = False
        if creature.id == SHARK:
            try:
                xp, yp = random.choice([pos
                            for pos in neighbours if neighbours[pos]!=EMPTY
                                                and neighbours[pos].id==FISH])
                creature.energy += 2
                self.grid[yp][xp].dead = True
                self.grid[yp][xp] = EMPTY
                moved = True
            except IndexError:
                pass

        if not moved:
            try:
                xp, yp = random.choice([pos
                            for pos in neighbours if neighbours[pos]==EMPTY])
                if creature.id != FISH:
                    creature.energy -= 1
                moved = True
            except IndexError:
                xp, yp = creature.x, creature.y

        if creature.energy < 0:
            creature.dead = True
            self.grid[creature.y][creature.x] = EMPTY
        elif moved:
            x, y = creature.x, creature.y
            creature.x, creature.y = xp, yp
            self.grid[yp][xp] = creature
            if creature.fertility >= creature.fertility_threshold:
                creature.fertility = 0
                self.spawn_creature(creature.id, x, y)
            else:
                self.grid[y][x] = EMPTY

    def evolve_world(self):
        random.shuffle(self.creatures)
        ncreatures = len(self.creatures)
        for i in range(ncreatures):
            creature = self.creatures[i]
            if creature.dead:
                continue
            self.evolve_creature(creature)

        self.creatures = [creature for creature in self.creatures
                                                if not creature.dead]

world = World()
world.populate_world()

fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
ax = fig.add_subplot(111)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
im = ax.imshow(world.get_world_image_array(), interpolation='nearest', cmap=cm)

def update(frame):
    world.evolve_world()
    im.set_array(world.get_world_image_array())
    ax.set_title(f"Chronon: {frame + 1}")
    return im,

animation = FuncAnimation(fig, update, frames=MAX_CHRONONS, repeat=False, blit=True)

plt.show()