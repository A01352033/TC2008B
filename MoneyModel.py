from mesa import Agent, Model
from mesa.time import RandomActivation

# For a jupyter notebook add the following line:
# matplotlib inline
# The below is needed for both notebooks and scripts
import matplotlib.pyplot as plt

class MoneyAgent(Agent):
    def __init__(self, id, model):
        super().__init__(id, model)
        self.wealth = 1

    def step(self):
        # The agent's step will go here.
        if self.wealth == 0:
            return
        
        other_agent = self.random.choice(self.model.schedule.agents)
        other_agent.wealth += 1
        self.wealth -= 1

class MoneyModel(Model):
    def __init__(self, num_agents):
        self.schedule = RandomActivation(self)

        for i in range(num_agents):
            agent = MoneyAgent(i, self)
            self.schedule.add(agent)

    def step(self):
        self.schedule.step()

# Visualization code (Values are hardcoded for simplicity)
NUM_AGENTS = 10
MAX_ITERATIONS = 10

model = MoneyModel(NUM_AGENTS)
for i in range(MAX_ITERATIONS):
    model.step()

agent_wealth = [agent.wealth for agent in model.schedule.agents]
plt.hist(agent_wealth)
plt.show()

