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

                layers:list[int]=[10,8,4,2],
                genotype:tuple[float,...]=[
                    0.1643,  -1.9121,   1.5944,   3.0084,   2.9300,  -1.3188,  -5.6435,
                   -1.1207,  -6.7068,  13.2495, -13.3776,   0.8839,  -8.9438,  -3.2684,
                   -2.8342,  -7.2338,  -1.1386,   2.0257, -10.1124,   2.1708,   3.8349,
                   -4.3874,   7.3665,   7.7275,  -2.3309,   2.6131,  -1.1979,  -0.3579,
                   -6.2896,   6.8338,   1.1960,  -1.8479,  -8.7828,  -3.2830,   6.7781,
                   -0.7022,  -6.0627,  -1.0542,   4.0895,   7.1702,  -5.6166,  -1.1680,
                   -1.4099,  -2.6469,  -5.4377,   0.1143,   0.3085,   8.6710,   0.1529,
                    9.5528, -14.4642,  -1.8137,   0.6463,  -6.1233,  -8.1975,   7.5079,
                   25.5651,   6.5156,  -4.0340,  -3.5805,  -7.4343,  -5.8375,  10.9606,
                  -20.5883,  -3.8097,   1.3978,   0.1682,   5.7081,   7.9531, -15.8373,
                   -1.5991,  -5.5244,   0.8788,  -3.6709,   4.2530,  -2.9938,   9.4248,
                   -2.5505,  -5.8964,   5.8206,  -3.8756,  -3.3751,   1.4252,  -1.7092,
                   -5.2979,  11.9375,   2.3161,  -5.3447,  -7.3847,   3.5262,  -0.9650,
                   -9.5526,  -6.8624,  -5.1511, -12.3318,  -3.8675,  -1.5227,   6.1460,
                  -11.0559,  -7.6943,   2.8572,  -9.0164,  -4.8923,  -8.8885, -18.8021,
                   -1.6255,  -2.9247,  -1.9842,  -6.7870,  -0.9132,  -9.4671,  -1.8929,
                   -5.2352,  -4.3952,  -4.4597,   0.3902,  11.5250,   1.0827, -10.5244,
                   -3.4773,   4.1247,   8.7506,  18.9527,  15.0209,  11.4850, -11.0210,
                    8.9729,  -9.2783,  15.2756, -13.9489,   6.2456,  -7.8413, -21.4626,
                   12.1025],
                activation_func:str="sigmoid",

                state:int=None
    ):
        super().__init__(layers, genotype, activation_func)

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

        batch_mode = surrounding.ndim == 2 and np.shape(surrounding) != (3,3)  # Check if input is batch (2D) or single (1D)

        if not batch_mode:
            surrounding = surrounding[np.newaxis, :]  # Convert to batch (1, 9)

        states,pheroA,pheroB = np.vectorize(lambda a: Cell.getAll(a))(surrounding)

        surr_pheros = np.where(np.expand_dims(self.state, axis=-1) == 0, pheroA, pheroB)
        surr_pheros = np.where(states == 3, -np.inf, surr_pheros)

        surr_pheros = surr_pheros.reshape(len(surr_pheros),9)

        neighbours = np.delete(surr_pheros, 4, axis=1)

        t_probs = np.full_like(neighbours,0.1)

        #// print(neighbours, t_probs, sep='\n', end='\n\n')

        # Calc favoured translation from current bearing
        x,y = np.round(np.cos(self.bearing)).astype(int), np.round(-np.sin(self.bearing)).astype(int)

        # 1D index of neighbours
        heading_index = (y + 1) * 3 + (x + 1)
        heading_index -= (heading_index >= 4) # account for deleted index 4

        # Favour the space that the agent is facing
        # if batch_mode:
        t_probs[np.arange(len(t_probs)), heading_index] += .2
        # else:
            # t_probs[heading_index] += .2

        # Fully discourage random exploration from selecting an out-of-bounds space
        t_probs[neighbours == -np.inf] = 0

        if np.random.rand() > 0:#dt/5:

            indices = np.argwhere(
                neighbours == np.amax(
                    np.expand_dims(neighbours,axis=-1),
                    axis=1
                )
            )

            #// t_probs_before = t_probs.copy()
            #// print(*zip(t_probs,indices), sep='\n')
            # if batch_mode:
            t_probs[indices[:,0], indices[:,1]] += .4
            # else:
            #     t_probs[indices] += .4


            index = np.argmax(t_probs, axis=1 if batch_mode else None)
            
            # index = np.argmax(neighbours)
        
        else:
            # Normalize probabilities for each batch
            t_probs /= t_probs.sum(axis=1, keepdims=True)  # Ensure each row sums to 1
            
            # Vectorized random choice (for batch processing)
            index = np.array([np.random.choice(8, p=t_probs[i]) for i in range(t_probs.shape[0])])

        # print(neighbours, t_probs, sep='\n', end='\n\n')

        index += (index >= 4) # Account for (1,1) being removed

        translation = np.stack(((index % 3) - 1, (index // 3) - 1), axis=1 if batch_mode else 0)
        
        if batch_mode:
            new_bearing = np.arctan2(-translation[:, 1], translation[:, 0]) % (2 * np.pi)
        else:
            new_bearing = np.arctan2(-translation[1], translation[0]) % (2 * np.pi)

        delta_bearing = new_bearing - self.bearing
        self.bearing = new_bearing

        #// print(translation, index)

        # Update position (only in single mode)
        # if not batch_mode:
        #     self.pos += np.clip(translation * dt * self.speed, -1, 1)
        #     return translation, delta_bearing  # Return single values

        return translation, delta_bearing  # Return batch results
    
    def get_pos(self):
        return np.array([int(a) for a in self.pos])

    def update(self, surrounding:list[float], dt:float=1):
        if time() - self.timer >= 10:
            self.state = not self.state
            self.timer = time()
            self.reward = 0x7FFFFFFF
            print("SWITCH!")

        inp = torch.cat([
            torch.Tensor([self.bearing]),
            torch.Tensor([self.state]),
            torch.DoubleTensor(
                np.delete([Cell.normalize(s) for surr in surrounding for s in surr], 4)
            )
        ])

        out = self.predict(inp).squeeze(0).detach().numpy().round(4)

        gt_out = self.follow_phero(surrounding, dt)
        b = (gt_out[1] + pi) % (2*pi) - pi
        act = np.array([max(b, 0), max(-b, 0)]).round(4)

        print(out, act)

        # self.pos += np.clip(gt_out[0] * dt * self.speed, -1, 1)

        # return



        l,r = out
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