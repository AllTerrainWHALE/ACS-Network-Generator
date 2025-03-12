from typing import Union
import numpy as np
import threading as th

from time import time, sleep
from math import sin, cos, pi, radians, degrees, sqrt
# from scipy.stats import norm

from src.cell import Cell
from src.utils import bcolors as bc

class Agent():

    learning_rate = 0.95
     
    def __init__(self,
                position:tuple[float,float]=(0,0),
                bearing:float=None,
                speed:float=5,

                state:int=None
    ):

        self.pos = np.array(position, dtype=np.float64) # (x,y)
        self.bearing = bearing if bearing else np.random.uniform(0,2*pi)
        self.speed = speed

        self.random_exploration = .2
        self.exploration_steps = 0
        self.max_exploration_steps = 0

        self.state = state if state != None else np.random.randint(0,2)

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

        if phero_val <= 10:
            returning = phero_val, self.state
            self.state = 0 #int(not self.state)
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

        neighbours = np.where(self.state == 0, pheroA, pheroB)
        neighbours = (neighbours - min(neighbours)) / (max(neighbours) - min(neighbours)) if max(neighbours) > 0 else np.zeros_like(neighbours)
        # neighbours = np.where(items == Cell.item.WALL, 0, neighbours)

        #_ Calc favoured translation from current bearing
        x,y = np.round(np.cos(self.bearing)).astype(int), np.round(-np.sin(self.bearing)).astype(int)

        #_ 1D index of neighbours
        heading_index = (y + 1) * 3 + (x + 1)
        heading_index -= (heading_index >= 4) # account for deleted index 4

        #_ Initialize equal probability of moving to a neighbouring cell
        t_probs = np.full_like(neighbours, 0.1, dtype=float)

        #_ Fully discourage from selecting an out-of-bounds space
        t_probs[items == Cell.item.WALL] = 0

        #_ Massively encourage following into nest/food source
        t_probs[items == (Cell.item.FOOD if self.state == 1 else Cell.item.NEST)] = 1 #// (Cell.item.FOOD if self.state == 1 else Cell.item.NEST)

        #_ Select which cell to move towards
        if self.exploration_steps == 0 and np.random.rand() > dt * self.random_exploration: #? Move to highest phero

            # Add pheromone values as probabilities to `t_probs`
            t_probs += neighbours / max(neighbours.sum(),1)

            # Favour the space that the agent is facing (if it's not a wall)
            if items[heading_index] != Cell.item.WALL:
                t_probs[heading_index] += .2

            # Get all indicies with the max value
            #//   and favour those in the probabilities array
            #// indices = np.argwhere(neighbours == np.amax(neighbours)).flatten()
            #// t_probs[indices] += .3

            # Select cell with the highest probability
            # if np.all(t_probs == np.empty_like(t_probs)): print(t_probs)
            try:
                max_indices = np.argwhere(t_probs == np.amax(t_probs)).flatten()
                index = np.random.choice(max_indices)
            except Exception as e:
                print(e,end='\n\n')
                print(max_indices, t_probs, neighbours)
        
        else: #? Random exploration
            self.exploration_steps += self.max_exploration_steps if self.exploration_steps == 0 else -1

            # Normalize probabilities
            t_probs /= t_probs.sum()  #? Ensure all probs sum to 1
            
            # Randomly select a direction
            index = np.random.choice(8, p=t_probs)

        # print(self.random_exploration)

        #// print(neighbours, t_probs, sep='\n', end='\n\n')

        #_ Adjust resulting index for removed [1, 1] index at the start
        index += int(index >= 4) # Account for (1,1) being removed

        #_ Find target translation and required delta bearing
        translation = (index % 3) - 1, (index // 3) - 1
        
        new_bearing = np.arctan2(-translation[1], translation[0]) #// + (2*pi)) % (2*pi)

        delta_bearing = new_bearing - self.bearing #// % (2*pi)

        return delta_bearing
    
    def get_pos(self):
        return np.array([int(a) for a in self.pos])

    def update(self, surrounding:"np.ndarray | list[float]", dt:float=1):
        dt = 1
        if self.switch_cd == None:
            self.switch_cd = time()

        # elif time() - self.switch_cd >= 10:
        #     self.state = 0
        #     self.switch_cd = time()
        #     self.reward = Cell.MAX_PHERO

        cell_states = np.vectorize(Cell.getItem)(surrounding)
        if np.all(cell_states != Cell.item.NONE):
            if np.all(cell_states == Cell.item.NEST):
                self.state = 1
            elif np.all(cell_states == Cell.item.FOOD):
                self.state = 0

            self.switch_cd = time()
            self.reward = Cell.MAX_PHERO

        #// direction = np.array((cos(self.bearing), -sin(self.bearing)))
        #// self.pos += np.clip(direction * dt * self.speed, -1, 1)

        #_ Restrict agent to environment boundaries
        # Bouncing off of top and bottom bounds
        # if any(np.array_equal(np.full(3, Cell.setState(0, 3)), edge) for edge in surrounding[(0,-1),:]):
        #     #? normal_angle = surface angle->(pi/2) + pi / 2 = pi
        #     self.bearing = (2 * pi) - self.bearing
        #     self.bearing %= (2*pi)

        # # Bouncing off of left and right bounds
        # elif any(np.array_equal(np.full(3, Cell.setState(0, 3)), edge) for edge in surrounding[:,(0,-1)].T):
        #     #? normal_angle = surface angle->(pi) + pi / 2 = pi * 1.5
        #     self.bearing = (2 * (pi*1.5)) - self.bearing
        #     self.bearing %= (2*pi)

        # print(degrees(self.bearing))

        # else: #? What is this???
        #     self.bearing += pi * 0 * dt

        #_ Adjust heading towards desired neighbouring cell
        delta_bearing = self.detect_phero(surrounding, dt)
        # target_bearing = (self.bearing + delta_bearing) % (2*pi)

        # w = 0
        # self.bearing = (w * self.bearing + (1-w) * target_bearing) % (2*pi)
        self.bearing += delta_bearing
        self.bearing %= (2*pi)

        #_ Move in the direction of heading
        direction = np.array((cos(self.bearing), -sin(self.bearing)))
        new_pos = self.pos + np.clip(direction * self.speed * dt, -1, 1)

        self.pos = new_pos