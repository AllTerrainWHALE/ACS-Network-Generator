import numpy as np

from math import sin, cos, sqrt, pi
from torch import FloatTensor

from src.environment import Environment
from src.agent import Agent
from src.cell import Cell

class Colony:

    reward = Cell.MAX_PHERO

    def __init__(self,
            environment:Environment,
            nest_pos:tuple[int,int],
            colony_size:int,

            radius:int=5
    ):
        self.env = environment

        self.pos = np.array(nest_pos)
        self.radius = radius

        # self.env.grid[(*self.pos,)] = Cell.setState(self.env.grid[(*self.pos,)], int(Cell.state.NEST))

        #_ Set environment states in a circle around the colony's position
        r,c = self.env.grid.shape
        y,x = np.ogrid[:r,:c]

        distance = (x - self.pos[0])**2 + (y - self.pos[1])**2 # distance^2 from circle center
        mask = distance <= self.radius**2

        self.env.grid[mask] = np.vectorize(lambda c: Cell.setItem(c, Cell.item.NEST))(self.env.grid[mask])

        self.agents = np.array([])
        for a in range(colony_size):
            r = radius * sqrt(np.random.uniform())
            theta = np.random.uniform() * 2 * pi

            x,y = self.pos[0] + r * cos(theta), self.pos[1] + r * sin(theta)

            self.agents = np.append(self.agents, Agent(
                position=(x,y), state=1 #(a-1)%2
            ))

    def update(self,dt:float=1):

        for a in self.agents:
            surr = self.env.p_grid[int(a.pos[0]):int(a.pos[0]+3), int(a.pos[1]):int(a.pos[1]+3)].T

            # Agent release pheromones
            phero_amount, phero_type = a.release_phero(surr)

            self.env.grid[(*a.get_pos(),)] = (int(self.env.grid[(*a.get_pos(),)]) & ~(Cell.MAX_PHERO << phero_type*Cell.MASK_STEP)) | ((int(phero_amount) & Cell.MAX_PHERO) << (phero_type*Cell.MASK_STEP))

            #// print(f"{phero_type}: {phero_amount} - {Cell.getAll(self.grid[(*a.get_pos(),)])}")

            # Agent move
            a.update(surr, dt)