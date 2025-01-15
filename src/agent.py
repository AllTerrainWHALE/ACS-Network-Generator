import numpy as np
import threading as th

from src.cell import Cell
from time import time, sleep
from math import sin, cos, pi

class Agent:
     
    def __init__(self,
                position:tuple[float,float]=(0,0),
                bearing:float=0,
                speed:float=40
    ):
        self.pos = np.array(position)
        self.bearing = pi/2
        self.speed = speed

    def release_phero(self):
        # return pheromone amount, and phero type (1 = A | 0 = B)
        return 0x7FFFFFFF, 1

    def update(self, surr:list[float], dt:float=1):

        # Bouncing off of top and bottom bounds
        if any(np.array_equal([-1, -1, -1], edge) for edge in (surr[:,0], surr[:,2])):
            # normal_angle = surface angle->(pi/2) + pi / 2 = pi
            self.bearing = 2 * pi - self.bearing
            self.bearing = self.bearing % (2*pi)

        # Bouncing off of left and right bounds
        elif any(np.array_equal([-1, -1, -1], edge) for edge in (surr[0,:], surr[2,:])):
            # normal_angle = surface angle->(pi) + pi / 2 = pi * 1.5
            self.bearing = 2 * (pi*1.5) - self.bearing
            self.bearing = self.bearing % (2*pi)

        else:
            self.bearing += pi * 0 * dt

        direction = np.array((cos(self.bearing), sin(self.bearing)))
        new_pos = self.pos + direction * self.speed * dt

        self.pos = new_pos