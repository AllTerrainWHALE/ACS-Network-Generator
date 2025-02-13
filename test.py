import numpy as np
import threading as th
import random

import torch
import torch.nn as nn

from math import pi, radians, degrees
from time import sleep
from datetime import datetime

from src.cell import Cell, State

#? Print statement to stop the first output being merged with shit in the terminal startup
#? IDK what the fuck goes on there
print()

#! Cell Class Testing
# c = 0

# c = Cell.setState(c, 1)
# c = Cell.setPheroB(c, 0x7FFFFFFF)
# c = Cell.setPheroA(c, 0x7FFFFFFF)

# print(Cell.getState(c))
# print(Cell.getPheroA(c))
# print(Cell.getPheroB(c))
# print(Cell.getAll(c))

# c = Cell.setPheroA(c, 0x7FFFFFFF)

# print(Cell.getState(c))
# print(Cell.getPheroA(c))
# print(Cell.getPheroB(c))
# print()
# print(0x7FFFFFFF)

#! Pheromone Dispersal and Evaporation Testing
# xy_phero = 0x7FFFFFFF
# for _ in range(2):
#     blur = xy_phero / 9

#     diffusionDelta = 0.01
#     evaporationDelta = 0.01 * 0x7FFFFFFF

#     diff = (diffusionDelta * xy_phero) + ((1-diffusionDelta) * blur)
#     diff_evap = max(0, diff - evaporationDelta)

#     print(f"{round(diff_evap, 3):,}")

#     xy_phero = diff_evap

#! Environment Padding
# env = np.random.randint(0,2,(5,5))
# pos = 0,3


# print(env)
# print()

# # padded_env = np.pad(env, pad_width=1, mode='constant', constant_values=-1)

# surr = np.pad(env, pad_width=1, mode='constant', constant_values=-1)[pos[0]:pos[0]+3, pos[1]:pos[1]+3]

# print(surr)
# print()

# if any(np.array_equal([-1, -1, -1], edge) for edge in (surr[0,:], surr[2,:], surr[:,0], surr[:,2])):
#     print('At edge!')

# if (surr[0,:] == [-1,-1,-1]).all(): print('Top')
# if (surr[2,:] == [-1,-1,-1]).all(): print('Bottom')
# if (surr[:,0] == [-1,-1,-1]).all(): print('Left')
# if (surr[:,2] == [-1,-1,-1]).all(): print('Right')

#! Agent Movement Selection Testing
# env = np.random.randint(0,10,(5,5))
# env[0,1] = Cell.setPheroA(env[0,1],0x7FFFFFFF)
# env[2,0] = Cell.setPheroA(env[2,0],0x7FFFFFFF)
# # poss = np.array([(1,2),(2,2),(0,1)])
# pos = np.array([2,2])
# bearing = pi*8/4
# # state = 0

# # Calc favoured translation from current bearing
# x,y = round(np.cos(bearing)), round(-np.sin(bearing))

# # 1D index of neighbours
# heading_index = (y + 1) * 3 + (x + 1)
# heading_index -= heading_index // 5 # account for deleted index 4

# print(heading_index)

# print(f"Environment:\n{env}", end='\n\n')

# surrounding = env[int(pos[1]-1):int(pos[1]+2), int(pos[0]-1):int(pos[0]+2)]
# print(f"Surrounding:\n{surrounding}", end='\n\n')

# neighbours = np.delete(surrounding, 4)
# print(f"Neighbours:\n{neighbours}", end='\n\n')

# print(env[:, (0,-1)])
# print()
# print(env[:, (0,-1)].T)
# print()
# print(np.amax(list(map(lambda a: a + 1, env))))

# bearing %= 2*pi

# section = int(bearing // (pi/8))

# heading_to_index = [5, 2, 2, 1, 1, 0, 0, 3, 3, 6, 6, 7, 7, 8, 8, 5]

# print(heading_to_index[section])

# print(f"Section Index: {section_index}")
# print(f"Heading Index: {heading_index}")

# e = np.delete(env, 4)

# indexs = np.argwhere(
#     e == np.amax(
#         e
#     )
# ).flatten()

                
# print(indexs)

# for index in indexs:
#     print('-'*10)

#     index += index // 4
#     print(index)
#     print()

#     move = (index % 3) - 1, (index // 3) - 1

#     print(move)
#     print(pos + move)

#! Numpy Conditional Indexing and Assignment
# arr1 = np.ones(10)
# arr2 = np.random.randint(0,2,10)

# print(arr1, arr2, sep='\n', end='\n\n')

# arr1[arr2 == 2] = 0

# print(arr1)

#! Threading Testing
# def func(num1, num2):
#     print(num1+num2)

# thread = th.Thread(target=func, args=(4,5,))

# thread.start()

# thread.join()

#! Circular Indexing Testing
# arr = np.zeros((10,10))
# pos = [5,5]
# radius = 3

# r,c = arr.shape
# y,x = np.ogrid[:r,:c]

# distance = (x - pos[0])**2 + (y - pos[1])**2 # distance^2 from circle center
# mask = distance <= radius**2

# arr[mask] = 1

# g = np.zeros((*arr.shape,3))

# g[arr==1] = arr[arr==1,np.newaxis] * np.array(((153,76,0)))

# print(g)

#! Neural Network Preditc Method Testing
# def new_genotype(layers, weight_bias_magnitude:float=.01):
#     # create genotype structured as:
#     #       [Ws_1,bs_1,Ws_2,bs_2,...,Ws_x,bs_x]
#     genotype = np.array([])
#     for i in range(1,len(layers)):
#         genotype = np.append(
#             genotype,
#             np.random.uniform(layers[i-1]*layers[i] + layers[i]) * weight_bias_magnitude
#         )
#     return torch.from_numpy(genotype)

# def predict(x,layers,genotype):
    
#     start_idx = 0
#     for i in range(1,len(layers)):
        
#         middle_idx = start_idx + len(x)*layers[i]
#         end_idx = middle_idx + layers[i]

#         Ws = genotype[start_idx:middle_idx]
#         bs = genotype[end_idx-layers[i]:end_idx]

#         Ws = torch.reshape(Ws, (len(x),layers[i]))

#         print(Ws, bs, sep=' | ')

#         x = torch.matmul(x,Ws) + bs

#         start_idx = end_idx

#     return x

# layers = [4,4,2]
# # input = torch.from_numpy(np.random.uniform(0,1,layers[0]))
# input = torch.rand(layers[0])
# genotype = new_genotype(layers)

# nn.Parameter

# print(genotype, end='\n\n')

# x = predict(input, layers, genotype)

#! Adding element to torch.Tensor
# print()
# rand_general = torch.cat([
#     torch.cat((
#         torch.rand(1) * 2*pi,               # Bearing
#         torch.randint(0,2,(1,)),            # State
#         torch.rand(8, dtype=torch.float64) * 0xFFFFFFFFFFFFFFFF  # Neighbouring cells
#     )).unsqueeze(0) for i in range(5)
# ])

# rand_empty_surr = torch.cat([
#     torch.cat((
#         torch.rand(1) * 2*pi,               # Bearing
#         torch.IntTensor([i%2]),             # State
#         torch.ones(8, dtype=torch.float64) * torch.rand(1, dtype=torch.float64) * 0xFFFFFFFFFFFFFFFF # Neighbouring cells
#     )).unsqueeze(0) for i in range(5)
# ])

# rand_all = torch.cat([rand_general, rand_empty_surr])

# # print(rand_general, rand_empty_surr, rand_all, sep='\n\n')

# print(list(map(Cell.getAll, rand_empty_surr[0])))

#! Bearing left and right testing
# init_bearing, final_bearing = 0, pi#np.random.uniform(0, 2*pi, 2)
# print(init_bearing, final_bearing)

# #// delta_dearing = final_bearing - init_bearing
# delta_bearing = (final_bearing - init_bearing + pi) % (2*pi) - pi

# left, right = max(delta_bearing,0), max(-delta_bearing,0)

# print(delta_bearing, (left,right), sep='\n')

#! PyTorch vectorizing testing
# random_states = 10

# inputs = torch.cat([
#     torch.cat((
#         torch.rand(1) * 2*pi,       # Bearing
#         torch.IntTensor([i%2]),     # State
#         torch.rand(8, dtype=torch.float64) * 0xFFFFFFFFFFFFFFFF  # Neighbouring cells
#     )).unsqueeze(0) for i in range(random_states)
# ])   # Shape: [random_states, 10]

# x_batch = inputs.clone()

# # Normalize neighbour cells
# print(x_batch[:,2:].shape)
# x_batch[:, 2:] = torch.vmap(lambda x: torch.vmap(Cell.normalize)(x))(x_batch[:,2:])

# print(x_batch[:,2:])

#! PyTorch Tensor shuffling
# random_states = 10

# rand_general = torch.cat([
#     torch.cat((
#         torch.rand(1) * 2*pi,       # Bearing
#         torch.IntTensor([i%2]),     # State
#         torch.rand(8, dtype=torch.float64) * 0xFFFFFFFFFFFFFFFF  # Neighbouring cells
#     )).unsqueeze(0) for i in range(int(random_states * 0.7))
# ])   # Shape: [random_states, 10]

# rand_empty_surr = torch.cat([
#     torch.cat((
#         torch.rand(1) * 2*pi,       # Bearing
#         torch.IntTensor([i%2]),     # State
#         torch.zeros(8, dtype=torch.float64) * torch.rand(1, dtype=torch.float64) * 0xFFFFFFFFFFFFFFFF # Neighbouring cells
#     )).unsqueeze(0) for i in range(int(random_states * 0.3))
# ])

# inputs = torch.cat([rand_general, rand_empty_surr])

# x_batch = inputs[torch.randperm(inputs.size()[0])]

# print(x_batch)
