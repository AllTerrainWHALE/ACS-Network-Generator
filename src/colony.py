import numpy as np

from math import sin, cos, sqrt, pi
from torch import FloatTensor

from src.environment import Environment
from src.agent import Agent
from src.cell import Cell

class Colony:

    reward = Cell.MAX_PHERO

    def __init__(self,
            environment:"Environment",
            nest_pos:tuple[int,int],
            colony_size:int,

            radius:int=5
    ):
        """
        Colony constructor. The nest node is placed in the environment, and agents are place randomly within the circle.

        :param Environment environment: The environment in which the colony exists.
        :param [int,int] nest_pos: The position of the colony in the environment.
        :param int colony_size: The number of agents in the colony.
        :param int radius: The radius of the colony.
        """
        self.env = environment

        self.pos = np.array(nest_pos)
        self.radius = radius

        #_ Set environment states in a circle around the colony's position
        r,c = self.env.grid.shape
        y,x = np.ogrid[:r,:c]

        distance = (x - self.pos[1])**2 + (y - self.pos[0])**2 # distance^2 from circle center
        mask = distance <= self.radius**2

        self.env.grid[mask] = np.vectorize(lambda c: Cell.setItem(c, Cell.item.NEST))(self.env.grid[mask])

        #_ Create and place agents randomly within the nest circle
        self.agents = np.array([])
        for a in range(colony_size):
            r = radius * sqrt(np.random.uniform())
            theta = np.random.uniform() * 2 * pi

            x,y = self.pos[1] + r * cos(theta), self.pos[0] + r * sin(theta)

            self.agents = np.append(self.agents, Agent(
                position=(y,x), state=1 #(a-1)%2
            ))

        np.random.choice(self.agents).tracked = True

    def update(self,dt:float=1):
        """
        Update the colonies agents.

        :param float dt: The time step for the update.
        """

        for a in self.agents:
            surr = self.env.p_grid[int(a.pos[0]):int(a.pos[0]+3), int(a.pos[1]):int(a.pos[1]+3)].T

            #_ Agent release pheromones
            phero_amount, phero_type = a.release_phero(surr)

            # Update pheromone values in the environment
            self.env.grid[(*a.get_pos(),)] = (int(self.env.grid[(*a.get_pos(),)]) & ~(Cell.MAX_PHERO << phero_type*Cell.MASK_STEP)) | ((int(phero_amount) & Cell.MAX_PHERO) << (phero_type*Cell.MASK_STEP))

            #_ Agent move
            a.update(surr, dt)