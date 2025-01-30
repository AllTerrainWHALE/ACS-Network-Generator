from typing import Union
import numpy as np
import threading as th

import torch

from time import time, sleep
from math import sin, cos, pi, radians, degrees

from src.agentnn import AgentNN
from src.cell import Cell

class Agent(AgentNN):

    learning_rate = 1
     
    def __init__(self,
                position:tuple[float,float]=(0,0),
                bearing:float=None,
                speed:float=5,

                layers:list[int]=[10,4,2],
                genotype:tuple[float,...]=None,

                state:int=None
    ):
        super().__init__(layers, genotype)

        self.pos = np.array(position, dtype=np.float64) # (x,y)
        self.bearing = bearing if bearing else np.random.uniform(0,2*pi)
        self.speed = speed

        self.state = state if state != None else np.random.randint(0,2)

        self.timer = time()

        self.reward = 0x7FFFFFFF

    # def new_genotype(self, weight_bias_magnitude:float=.01):
    #     # create genotype structured as:
    #     #       [Ws_1,bs_1,Ws_2,bs_2,...,Ws_x,bs_x]
    #     genotype = np.array([])
    #     for i in range(1,len(self.layers)):
    #         genotype = np.append(
    #             genotype,
    #             np.random.uniform(low=0,high=1,size=self.layers[i-1]*self.layers[i] + self.layers[i]) * weight_bias_magnitude
    #         )
    #     return genotype

    def release_phero(self, surrounding):
        # return pheromone amount, and phero type (1 = A | 0 = B)

        surr_pheroX = np.vectorize(lambda s: Cell.getPheroA(s) if self.state == 1 else Cell.getPheroB(s))(surrounding)
        
        neighbours = np.delete(surr_pheroX, 4)

        reward = self.reward #+ surr_pheroX[1,1]
        if Cell.getState(surrounding[1,1]) in [1,2]: pass
            #! reward = 0x7FFFFFFF
            #! self.state = not self.state

        phero_val = np.uint32(reward + Agent.learning_rate * np.amax(neighbours))
        phero_val = max(surr_pheroX[1,1], min(0x7FFFFFFF, phero_val))

        self.reward = 0

        return phero_val, self.state
    
    def follow_phero(self, surrounding:np.ndarray, dt:float=1):
        # if time() - self.timer >= 10:
        #     self.state = not self.state
        #     self.timer = time()
        #     self.reward = 0x7FFFFFFF
        #     print("SWITCH!")
        batch_mode = surrounding.ndim == 2  # Check if input is batch (2D) or single (1D)

        if not batch_mode:
            surrounding = surrounding[np.newaxis, :]  # Convert to batch (1, 9)

        states,pheroA,pheroB = np.vectorize(lambda a: Cell.getAll(a))(surrounding)

        surr_pheros = np.where(np.expand_dims(self.state, axis=-1) == 0, pheroA, pheroB)
        surr_pheros = np.where(states == 3, -np.inf, surr_pheros)

        neighbours = np.delete(surr_pheros, 4, axis=1)

        t_probs = np.full_like(neighbours,0.1)

        #// print(neighbours, t_probs, sep='\n', end='\n\n')

        # Calc favoured translation from current bearing
        x,y = np.round(np.cos(self.bearing)).astype(int), np.round(-np.sin(self.bearing)).astype(int)

        # 1D index of neighbours
        heading_index = (y + 1) * 3 + (x + 1)
        heading_index -= (heading_index >= 4) # account for deleted index 4

        # Favour the space that the agent is facing
        t_probs[np.arange(len(t_probs)), heading_index] += 0.2

        # Fully discourage random exploration from selecting an out-of-bounds space
        t_probs[neighbours == -np.inf] = 0

        if np.random.rand() > 0:#dt/5:

            indices = np.argwhere(
                neighbours == np.amax(
                    np.expand_dims(neighbours,axis=-1),
                    axis=1
                )
            )

            t_probs_before = t_probs.copy()

            t_probs[indices[:,0], indices[:,1]] += 0.4

            #// print(*zip(t_probs_before,t_probs,indices), sep='\n')

            index = np.argmax(t_probs, axis=1)
            
            # index = np.argmax(neighbours)
        
        else:
            # Normalize probabilities for each batch
            t_probs /= t_probs.sum(axis=1, keepdims=True)  # Ensure each row sums to 1
            
            # Vectorized random choice (for batch processing)
            index = np.array([np.random.choice(8, p=t_probs[i]) for i in range(t_probs.shape[0])])

        # print(neighbours, t_probs, sep='\n', end='\n\n')

        index += (index >= 4) # Account for (1,1) being removed

        translation = np.stack(((index % 3) - 1, (index // 3) - 1), axis=1)
        new_bearing = np.arctan2(-translation[:, 1], translation[:, 0]) % (2 * np.pi)

        delta_bearing = new_bearing - self.bearing
        self.bearing = new_bearing

        #// print(translation, index)

        # Update position (only in single mode)
        if not batch_mode:
            self.pos += np.clip(translation[0] * dt * self.speed, -1, 1)
            return translation[0], delta_bearing[0]  # Return single values

        return translation, delta_bearing  # Return batch results
    
    def get_pos(self):
        return np.array([int(a) for a in self.pos])

    def update(self, surrounding:list[float], dt:float=1):

        inp = torch.cat([
            torch.Tensor([self.bearing]),
            torch.Tensor([self.state]),
            torch.DoubleTensor(
                np.delete([Cell.normalize(s) for surr in surrounding for s in surr], 4)
            )
        ])

        l,r = self.predict(inp) * dt
        self.bearing += l
        self.bearing -= r


        # # Bouncing off of top and bottom bounds
        # if any(np.array_equal([-1, -1, -1], edge) for edge in surr[(0,-1),:]):
        #     # normal_angle = surface angle->(pi/2) + pi / 2 = pi
        #     self.bearing = 2 * pi - self.bearing
        #     self.bearing = self.bearing % (2*pi)

        # # Bouncing off of left and right bounds
        # elif any(np.array_equal([-1, -1, -1], edge) for edge in surr[:,(0,-1)].T):
        #     # normal_angle = surface angle->(pi) + pi / 2 = pi * 1.5
        #     self.bearing = 2 * (pi*1.5) - self.bearing
        #     self.bearing = self.bearing % (2*pi)

        # else:
        #     self.bearing += pi * 0 * dt

        direction = np.array((cos(self.bearing), sin(self.bearing)))
        new_pos = self.pos + np.clip(direction * self.speed * dt, -1, 1)

        self.pos = new_pos