import numpy as np
import threading as th

from src.cell import Cell
from time import time, sleep
from math import sin, cos, pi
from random import random

class Agent:
     
    def __init__(self,
                position:tuple[float,float]=(0,0),
                bearing:float=None,
                speed:float=5,

                state:int=None
    ):
        self.pos = np.array(position)
        self.bearing = bearing if bearing else np.random.uniform(0,2*pi)
        self.speed = speed

        self.state = state if state != None else 1

        self.reward = 0x7FFFFFFF

    def release_phero(self, surr):
        # return pheromone amount, and phero type (1 = A | 0 = B)

        learning_rate = 0.9 * 0x7FFFFFFF

        phero_val = (self.reward + surr[1,1]) + learning_rate * np.amax(list(map(Cell.getPheroA if self.state == 1 else Cell.getPheroB, surr)))

        # print(phero_val)

        self.reward = 0

        return int(phero_val), self.state
    
    def follow_phero(self, surr):
        pass
    
    def get_pos(self):
        return np.array([int(a) for a in self.pos])

    def update(self, surr:list[float], dt:float=1):

        # Bouncing off of top and bottom bounds
        if any(np.array_equal([-1, -1, -1], edge) for edge in surr[(0,-1),:]):
            # normal_angle = surface angle->(pi/2) + pi / 2 = pi
            self.bearing = 2 * pi - self.bearing
            self.bearing = self.bearing % (2*pi)

        # Bouncing off of left and right bounds
        elif any(np.array_equal([-1, -1, -1], edge) for edge in surr[:,(0,-1)].T):
            # normal_angle = surface angle->(pi) + pi / 2 = pi * 1.5
            self.bearing = 2 * (pi*1.5) - self.bearing
            self.bearing = self.bearing % (2*pi)

        else:
            self.bearing += pi * 0 * dt

        direction = np.array((cos(self.bearing), sin(self.bearing)))
        new_pos = self.pos + direction * self.speed * dt

        self.pos = new_pos