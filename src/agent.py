import numpy as np
import threading as th

from src.cell import Cell
from time import time, sleep
from math import sin, cos, pi, radians, degrees
from random import random

class Agent:

    learning_rate = 0.9
    heading_to_index = [5, 2, 2, 1, 1, 0, 0, 3, 3, 6, 6, 7, 7, 8, 8, 5]
     
    def __init__(self,
                position:tuple[float,float]=(0,0),
                bearing:float=None,
                speed:float=5,

                state:int=None
    ):
        self.pos = np.array(position, dtype=np.float64)
        self.bearing = bearing if bearing else 0 #np.random.uniform(0,2*pi)
        self.speed = speed

        self.state = state if state != None else 1

        self.reward = 0x7FFFFFFF

    def release_phero(self, surrounding):
        # return pheromone amount, and phero type (1 = A | 0 = B)
        
        neighbours = np.delete(surrounding, 4)

        phero_val = (
            self.reward +
            Agent.learning_rate *
            np.amax(
                list(
                    map(
                        Cell.getPheroA if self.state == 1 else Cell.getPheroB, neighbours
                    )
                )
            )
        )

        self.reward = phero_val

        return int(phero_val) if self.state == 1 else 0, self.state
    
    def follow_phero(self, surrounding, dt:float=1):
        surrounding = list(map(Cell.getPheroA if self.state == 0 else Cell.getPheroB, surrounding))

        neighbours = np.delete(surrounding, 4)

        if np.random.rand() > 0.1:

            # index = np.random.choice(
            #     np.argwhere(
            #         neighbours == np.amax(
            #             neighbours
            #         )
            #     ).flatten()
            # )
            index = np.argmax(neighbours)
        
        else:
            index = np.random.randint(0,8)
            print('> Random Move!')

        index += index // 4 # Account for (1,1) being removed

        move = ((index // 3) - 1) * dt * self.speed, ((index % 3) - 1) * dt * self.speed

        self.pos += np.clip(move, -1, 1)
    
    def get_pos(self):
        return np.array([int(a) for a in self.pos])

    def update(self, surr:list[float], dt:float=1):

        self.bearing += radians(10) * dt

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