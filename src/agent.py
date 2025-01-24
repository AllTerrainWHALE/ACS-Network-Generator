import numpy as np
import threading as th

from time import time, sleep
from math import sin, cos, pi, radians, degrees

from src.cell import Cell

class Agent:

    learning_rate = 1
     
    def __init__(self,
                position:tuple[float,float]=(0,0),
                bearing:float=None,
                speed:float=5,

                state:int=None
    ):
        self.pos = np.array(position, dtype=np.float64) # (x,y)
        self.bearing = bearing if bearing else np.random.uniform(0,2*pi)
        self.speed = speed

        self.state = state if state != None else np.random.randint(0,2)

        self.timer = time()

        self.reward = 0x7FFFFFFF

    def release_phero(self, surrounding):
        # return pheromone amount, and phero type (1 = A | 0 = B)

        surr_pheroX = np.vectorize(lambda s: Cell.getPheroA(s) if self.state == 1 else Cell.getPheroB(s))(surrounding)
        
        neighbours = np.delete(surr_pheroX, 4)

        reward = self.reward #+ surr_pheroX[1,1]
        if Cell.getState(surrounding[1,1]) in [1,2]:
            reward = 0x7FFFFFFF
            #! self.state = not self.state

        phero_val = np.uint32(reward + Agent.learning_rate * np.amax(neighbours))
        phero_val = max(surr_pheroX[1,1], min(0x7FFFFFFF, phero_val))

        self.reward = 0

        return phero_val, self.state
    
    def follow_phero(self, surrounding, dt:float=1):
        if time() - self.timer >= 10:
            self.state = not self.state
            self.timer = time()
            self.reward = 0x7FFFFFFF
            print("SWITCH!")


        # surrounding = list(map(Cell.getPheroA if self.state == 0 else Cell.getPheroB, surrounding))
        states,pheroA,pheroB = np.vectorize(lambda a: Cell.getAll(a))(surrounding)

        surr_pheros = pheroA if self.state == 0 else pheroB
        surr_pheros = np.where(states == 3, -np.inf, surr_pheros)

        neighbours = np.delete(surr_pheros, 4)

        t_probs = np.full(8,0.1)

        #// print(neighbours, t_probs, sep='\n', end='\n\n')

        # Calc favoured translation from current bearing
        x,y = round(np.cos(self.bearing)), round(-np.sin(self.bearing))

        # 1D index of neighbours
        heading_index = (y + 1) * 3 + (x + 1)
        heading_index -= heading_index // 5 # account for deleted index 4

        # Favour the space that the agent is facing
        t_probs[heading_index] += 0.2

        # Fully discourage random exploration from selecting an out-of-bounds space
        t_probs[neighbours == -np.inf] = 0

        if np.random.rand() > dt/5:

            # index = np.random.choice(
            indices = np.argwhere(
                neighbours == np.amax(
                    neighbours
                )
            ).flatten()
            # )
            t_probs[indices] += 0.4

            index = np.argmax(t_probs)
            
            # index = np.argmax(neighbours)
        
        else:
            # t_probs = np.where(t_probs!=0, 0.1, 0)
            index = np.random.choice(np.arange(8), p=t_probs/np.sum(t_probs))
            #// print('> Random Move!')

        # print(neighbours, t_probs, sep='\n', end='\n\n')

        index += index // 4 # Account for (1,1) being removed

        translation = np.array(((index % 3) - 1, (index // 3) - 1))
        new_bearing = np.arctan2(-translation[1], translation[0]) % (2*pi)

        self.bearing = new_bearing

        #// print(translation, index)

        self.pos += np.clip(translation * dt * self.speed, -1, 1)
    
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