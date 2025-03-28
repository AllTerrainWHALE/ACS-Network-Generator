from typing import Union
import numpy as np
import threading as th

from time import time, sleep
from math import sin, cos, pi, radians, degrees, sqrt
# from scipy.stats import norm

from src.cell import Cell
from src.utils import bcolors as bc

class Agent():

    learning_rate = 0.9
     
    def __init__(self,
                position:tuple[float,float]=(0,0),
                bearing:float=None,
                speed:float=5,

                state:int=None
    ):

        self.pos = np.array(position, dtype=np.float64) # (x,y)
        self.bearing = bearing if bearing else np.random.uniform(0,2*pi)
        self.speed = speed

        self.p_t = .3 # Popularity Threshold
        self.alpha = 0.7
        self.beta = 0.1

        self.cont_search_prob = .2

        self.exploration_prob = .5
        self.exploration_steps = 0
        self.max_exploration_steps = 5

        self.searching = 0

        self.state = state if state != None else np.random.randint(0,2)
        
        self.problem_child = False
        self.tracked = False
        self.lost = False

        self.switch_cd = None # Phero following switch cooldown

        self.reward = Cell.MAX_PHERO

    def release_phero(self, surrounding):
        # return pheromone amount, and phero type (1 = A | 0 = B)

        surr_pheroX = np.vectorize(lambda s: Cell.getPheroA(s) if self.state == 1 else Cell.getPheroB(s))(surrounding)
        
        neighbours = np.delete(surr_pheroX, 4)

        reward = self.reward #+ surr_pheroX[1,1]

        phero_val = np.uint32(reward + Agent.learning_rate * np.amax(neighbours))
        phero_val = max(surr_pheroX[1,1], min(Cell.MAX_PHERO, phero_val))

        self.reward = 0

        if phero_val <= 500 and not self.lost:
            returning = phero_val, self.state
            self.state = int(not self.state)
            self.lost = True
            self.switch_cd = time()
            return returning

        return phero_val, self.state
    
    def detect_phero(self, surrounding:np.ndarray, dt:float=1):
        if type(surrounding) != np.ndarray and type(surrounding) == list:
            surrounding = np.array(surrounding, dtype=np.uint64)

        assert type(surrounding) == np.ndarray, f"Invalid parameter type of `{type(surrounding)}` for `surrounding`. Must be of either type `np.ndarray` or `list`"

        #_ Delete center cell (the agent position)
        neighbours = np.delete(surrounding, 4)

        #_ Retrieve properties of each cell
        items,pheroA,pheroB = np.vectorize(lambda a: Cell.getAll(a))(neighbours)
        pheroA, pheroB = (pheroA/Cell.MAX_PHERO).astype(np.float64), (pheroB/Cell.MAX_PHERO).astype(np.float64)

        pheroX,pheroY = np.where(self.state == 0, (pheroA, pheroB), (pheroB, pheroA))

        weights = np.clip(np.where(pheroY < self.alpha,
            pheroX-np.pow(pheroY, self.p_t/pheroY)+pheroX*self.beta*np.sin((pi/self.alpha)*pheroY),
            pheroX-np.pow(pheroY, self.p_t/pheroY)
        ), a_min=0, a_max=1)

        weights = (weights - min(weights)) / (max(weights) - min(weights)) if max(weights) > min(weights) else np.zeros_like(weights)
        weights = np.where(items == Cell.item.WALL, 0, weights)

        #_ Calc favoured translation from current bearing
        x,y = np.round(np.cos(self.bearing)).astype(int), np.round(-np.sin(self.bearing)).astype(int)

        #_ 1D index of neighbours
        heading_index = (y + 1) * 3 + (x + 1)
        heading_index -= (heading_index >= 4) # account for deleted index 4

        #_ Initialize equal probability of moving to a neighbouring cell
        t_probs = np.full_like(weights, 0.1, dtype=float)

        #_ Fully discourage from selecting an out-of-bounds space
        t_probs[items == Cell.item.WALL] = 0

        #_ Massively encourage following into nest/food source
        if time() - self.switch_cd > 10:
            mask = (
                np.where( # if, then
                    items == Cell.item.FOOD, True,
                np.where( # elif, then
                    items == Cell.item.NEST, True,
                False # else then
            )))
            t_probs[mask] += 2

        #_ Select which cell to move towards
        if self.exploration_steps == 0 and np.random.rand() > dt * self.exploration_prob: #? Move to highest phero

            # Add pheromone values as probabilities to `t_probs`
            t_probs += weights / max(weights.sum(),1)

            # Favour the space that the agent is facing (if it's not a wall)
            if items[heading_index] != Cell.item.WALL:
                t_probs[heading_index] += .05

            # Select cell with the highest probability
            max_indices = np.argwhere(t_probs == np.amax(t_probs)).flatten()
            index = np.random.choice(max_indices)
                
            if np.all(items == Cell.item.FOOD) and self.tracked:
                print(f'{"A" if self.state == 0 else "B"}:', list(zip(np.round(pheroA,3),np.round(pheroB,3))), index)
        
        else: #? Random exploration
            self.exploration_steps += self.max_exploration_steps if self.exploration_steps == 0 else -1

            # Normalize probabilities
            t_probs /= t_probs.sum()  #? Ensure all probs sum to 1
            
            # Randomly select a direction
            index = np.random.choice(8, p=t_probs)

        #_ Adjust resulting index for removed [1, 1] index at the start
        index += int(index >= 4) # Account for (1,1) being removed

        #_ Find target translation and required delta bearing
        translation = (index % 3) - 1, (index // 3) - 1
        
        new_bearing = np.arctan2(-translation[1], translation[0])

        delta_bearing = new_bearing - self.bearing

        return delta_bearing
    
    def get_pos(self):
        return np.array([int(a) for a in self.pos])

    def update(self, surrounding:"np.ndarray | list[float]", dt:float=1):
        assert type(surrounding) in [np.ndarray, list], f"Invalid parameter type of `{type(surrounding)}` for `surrounding`. Must be of either type `np.ndarray` or `list`"
        assert type(dt) == float, f"Invalid parameter type of `{type(dt)}` for `dt`. Must be of type `float`"

        if self.switch_cd == None:
            self.switch_cd = time()

        elif time() - self.switch_cd >= 60:
            self.state = int(not self.state)
            self.lost = True
            self.reward = 0
            self.searching = False
            self.switch_cd = time()

        items,pheroA,pheroB = np.vectorize(Cell.getAll)(surrounding)
        if np.all(items != Cell.item.NONE) and time() - self.switch_cd > 10:
            self.lost = False

            if np.all(items == Cell.item.NEST):
                self.state = 1
                self.searching = False
                self.reward = Cell.MAX_PHERO

            elif np.all(items == Cell.item.FOOD):
                # self.state = 0 if self.state == 1 else 1
                if np.any(pheroA > 0) and np.any(pheroB > 0):
                    self.state = 0 if np.average(pheroB) > np.average(pheroA) else 1
                elif np.any(pheroA > 0):
                    self.state = 0
                elif np.any(pheroB > 0):
                    self.state = 1


                self.reward = Cell.MAX_PHERO #- (np.average(pheroA) if self.state == 1 else np.average(pheroB))

            if self.tracked: print(f'State changed to {"A" if self.state == 0 else "B"}')

            self.switch_cd = time()

        #//_ Restrict agent to environment boundaries
        #// Bouncing off of top and bottom bounds
        #// if any(np.array_equal(np.full(3, Cell.setState(0, 3)), edge) for edge in surrounding[(0,-1),:]):
        #//     #? normal_angle = surface angle->(pi/2) + pi / 2 = pi
        #//     self.bearing = (2 * pi) - self.bearing
        #//     self.bearing %= (2*pi)
        #//
        #// # Bouncing off of left and right bounds
        #// elif any(np.array_equal(np.full(3, Cell.setState(0, 3)), edge) for edge in surrounding[:,(0,-1)].T):
        #//     #? normal_angle = surface angle->(pi) + pi / 2 = pi * 1.5
        #//     self.bearing = (2 * (pi*1.5)) - self.bearing
        #//     self.bearing %= (2*pi)

        #_ Adjust heading towards desired neighbouring cell
        delta_bearing = self.detect_phero(surrounding, dt)

        self.bearing += delta_bearing
        self.bearing %= (2*pi)

        #_ Move in the direction of heading
        direction = np.array((cos(self.bearing), -sin(self.bearing)))
        new_pos = self.pos + np.clip(direction * self.speed * dt, -1, 1)

        self.pos = new_pos