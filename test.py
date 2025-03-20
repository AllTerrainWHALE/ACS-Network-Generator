# %%
import numpy as np
import threading as th
import random

from math import pi, radians, degrees, sqrt, ceil, floor
from time import sleep
from datetime import datetime

from src.cell import Cell
from src.agent import Agent
from src.environment import Environment
from src.utils import *

#?// Print statement to stop the first output being merged with shit in the terminal startup
#?// IDK what the fuck goes on there
#// print()

# %% #! Cell Class Testing
c = 0

c = Cell.setItem(c, 1)
c = Cell.setPheroB(c, Cell.MAX_PHERO)
c = Cell.setPheroA(c, Cell.MAX_PHERO)

print(Cell.getItem(c))
print(Cell.getPheroA(c))
print(Cell.getPheroB(c))
print(Cell.getAll(c))

c = Cell.setPheroA(c, int(Cell.MAX_PHERO/2))

print(Cell.getItem(c))
print(Cell.getPheroA(c))
print(Cell.getPheroB(c))
print()
print(Cell.MAX_PHERO)
# %%
items = np.array([
    [0,0,1],
    [2,0,0],
    [0,0,0]
])

print(np.where(items == 1, True, np.where(items == 2, True, False)))

#%%

#! Pheromone Dispersal and Evaporation Testing
xy_phero = Cell.MAX_PHERO
dt = 1e-1
print(f"{xy_phero:,}")
for _ in range(10):
    blur = xy_phero / 9

    diffusionDelta = 0.7 * dt
    evaporationDelta = 0.1 * dt

    diff_evap = (xy_phero + diffusionDelta * (blur - xy_phero)) * (1 - evaporationDelta)
    

    print(f"{round(diff_evap/Cell.MAX_PHERO, 3):,}")

    xy_phero = diff_evap

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

# print(rand_general, rand_empty_surr, rand_all, sep='\n\n')

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

#! Pheromon excluding testing
# val = 0
# val = Cell.setAll(0, 3, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)

# print(f'{bin(val):>66}', '=', val)
# print(f'{bin(0x1FFFFFFFF):>66}', '=', 0x1FFFFFFFF)
# print(f'{bin(Cell.excludeState(val)):>66}', '=', Cell.excludeState(val))
# print(f'{bin(Cell.excludePheroA(val)):>66}', '=', Cell.excludePheroA(val))
# print(f'{bin(Cell.excludePheroB(val)):>66}', '=', Cell.excludePheroB(val))

#! Sample creating testing
# sample = Environment([3,3])
# # sample.grid[np.random.randint(0,3), np.random.randint(0,3)] = Cell.setPheroA(0, 1000)#np.random.randint(1000))
# sample.grid[2,2] = Cell.setPheroA(0, 1000)

# print(np.vectorize(Cell.getPheroA)(sample.grid), end='\n\n')

# for _ in range(10):
#     sample.disperseAndEvaporate()

#     print(np.vectorize(Cell.getPheroA)(sample.grid), end='\n\n')

# print(*np.vectorize(Cell.getAll)(Environment.sample_state()), sep='\n\n')


# samples = 5
# grid_samples_edge = sqrt(samples)
# grid_cell_edge = ceil(3 * grid_samples_edge)

# print(grid_samples_edge, grid_cell_edge)








    # @staticmethod
    # def sample_state(env:"Environment"=None):
    #     if env == None:
    #         env = Environment([3,3])

    #     assert env.grid.shape == (3,3), f"Sample environment must have a resolution of (3, 3), not {env.grid.shape}"

    #     index = np.random.randint(0,3), np.random.randint(0,3)

    #     while True:
    #         items = np.random.choice([0, 1, 2, 3], size=(3, 3), p=[0.88, 0.01, 0.01, 0.1])
    #         if np.any(np.delete(items,4) != State.WALL):
    #             env.grid = items.astype(np.uint64) << 62
    #             break
        
    #     env.grid[index] = Cell.setPheroA(env.grid[index], np.random.randint(0x7FFFFFFF))
    #     env.grid[index] = Cell.setPheroB(env.grid[index], np.random.randint(0x7FFFFFFF))

    #     for _ in range(np.random.randint(10)):
    #         env.disperseAndEvaporate()

    #     return env.grid



#! Old training samples generation
        # Prepare random training conditions
        # rand_general = torch.cat([
        #     torch.cat((
        #         (lambda b: torch.FloatTensor([np.sin(b),np.cos(b)]))(torch.rand(1) * 2*pi),       # Bearing
        #         torch.IntTensor([i%2]),     # State
        #         torch.rand(8, dtype=torch.float64) * 0xFFFFFFFFFFFFFFFF  # Neighbouring cells
        #     )).unsqueeze(0) for i in range(int((random_states) * 0.5))
        # ])   # Shape: [random_states, 10]

        # rand_equal_surr = torch.cat([
        #     torch.cat((
        #         (lambda b: torch.FloatTensor([np.sin(b),np.cos(b)]))(torch.rand(1) * 2*pi),       # Bearing
        #         torch.IntTensor([i%2]),     # State
        #         torch.ones(8, dtype=torch.float64) * torch.rand(1, dtype=torch.float64) * 0xFFFFFFFFFFFFFFFF # Neighbouring cells
        #     )).unsqueeze(0) for i in range(int((random_states) * 0.1))
        # ])

        # empty_surr = torch.cat([
        #     torch.cat([
        #         (lambda b: torch.FloatTensor([np.sin(b),np.cos(b)]))(torch.rand(1) * 2*pi), # Bearing
        #         torch.IntTensor([i%2]),    # State
        #         torch.zeros(8, dtype=torch.float64)              # Neighbouring cells
        #     ]).unsqueeze(0) for i in range(int((random_states) * 0.4))
        # ])

        # inputs = torch.cat([rand_general, rand_equal_surr, empty_surr]).to(AgentNN.device)
        # inputs = inputs[torch.randperm(inputs.size(0))]


#! Check pheros are being selected correctly
# inp = torch.cat([
#         torch.tensor([Cell.setAll(0, 3, 10, 5) for _ in range(8)], dtype=torch.int64).unsqueeze(0) for i in range(5)
# ])

# print(inp.dtype)
# print(inp)
# exit()

# agent_states = inp[:,2].numpy()
# surr = inp[:, 3:]

# states,pheroA,pheroB = np.vectorize(lambda a: Cell.getAll(a))(surr)
# print(*list(zip(states,pheroA,pheroB)), sep='\n\n')

# # surr = np.vectorize(lambda a: Cell.setState(a,3))(surr)

# # states,pheroA,pheroB = np.vectorize(lambda a: Cell.getAll(a))(surr)
# # print(*list(zip(states,pheroA,pheroB)), sep='\n\n')


# surr_pheros = np.where(np.expand_dims(agent_states, axis=-1) == 0, pheroA, pheroB)

# print(*list(zip(agent_states,surr_pheros)), sep='\n\n')


# val = torch.tensor(Cell.setAll(0, 3, 0x3FFFFFFF, 0x3FFFFFFF), dtype=torch.int64)
# print(f"0b{'-'*64}")
# print(bin(val.item()))
# print(int('1'*62, 2))
# print(val.item(), sep='\n')

# print(Cell.getAll(val.item()))

# print(hex(int('1'*32,2)))
# print(len(str(bin(2147483647))))

#! Ensure environment samples are being generated correctly
# random_states = 10

# env = Environment([3,3])
# simul_samples = torch.cat([
#     torch.tensor(
#         np.delete(Environment.sample_state(env),4), dtype=torch.int64
#     ).unsqueeze(0) for i in range(int(random_states*0.7))
# ])   # Shape: [random_states, 10]

# empty_samples = torch.cat([
#     torch.zeros(8, dtype=torch.int64)
#         .unsqueeze(0) for i in range(int((random_states) * 0.3))
# ])
# del env

# env_samples = torch.cat([simul_samples, empty_samples])
# env_samples = env_samples[torch.randperm(env_samples.size(0))]

# states,pheroA,pheroB = np.vectorize(lambda a: Cell.getAll(a))(env_samples)
# print(*list(zip(states,pheroA,pheroB)), sep='\n\n')


#! OLD SAMPLE GENERATION
    #? issue with dtyping in tensors, as you cannot have multiple different types in one tensor

        # sim_surr = torch.cat([
        #     torch.cat((
        #         (lambda b: torch.FloatTensor([np.sin(b),np.cos(b)]))(torch.rand(1) * 2*pi),       # Bearing
        #         torch.IntTensor([i%2]),     # Agent State
        #         torch.DoubleTensor(np.delete(Environment.sample_state(sample_env),4))
        #     )).unsqueeze(0) for i in range(int(random_states*0.7))
        # ])   # Shape: [random_states, 10]

        # empty_surr = torch.cat([
        #     torch.cat([
        #         (lambda b: torch.FloatTensor([np.sin(b),np.cos(b)]))(torch.rand(1) * 2*pi), # Bearing
        #         torch.IntTensor([i%2]),    # State
        #         torch.zeros(8, dtype=torch.float64)              # Neighbouring cells
        #     ]).unsqueeze(0) for i in range(int((random_states) * 0.3))
        # ])

# random_states = 10

# #_ Generate environment samples to train on
# random_states = 10

# env = Environment([3,3])
# simul_samples = torch.cat([ #? Shape: [random_states*0.7, 8]
#     torch.tensor(
#         np.delete(Environment.sample_state(env,0.5),4), dtype=torch.int64 #//np.full([3,3], Cell.setAll(0,3,10,5), dtype=np.int64)
#     ).unsqueeze(0) for i in range(random_states)
# ])
# del env

# # empty_samples = torch.cat([ #? Shape: [random_states*0.3, 8]
# #     torch.zeros(8, dtype=torch.int64)
# #         .unsqueeze(0) for i in range(int((random_states) * 0.3))
# # ])

# # env_samples = torch.cat([simul_samples, empty_samples])
# # env_samples = env_samples[torch.randperm(env_samples.size(0))]
# #         #? Shape: [random_states, 8]
# env_samples = simul_samples.to(AgentNN.device)
#         #? Shape: [random_states, 8]

# #_ Generate random bearings
# bearings = torch.cat([ #? Shape: [random_states, 2]
#     (lambda b: torch.FloatTensor([np.sin(b),np.cos(b)]))(torch.rand(1) * 2*pi)
#         .unsqueeze(0) for i in range(random_states)
# ])

# #_ Generate uniformly split states
# a_states = torch.cat([ #? Shape: [random_states]
#     torch.ByteTensor([i%2])
#         for i in range(random_states)
# ])


# #// print(*list(zip(env_samples,bearings,a_states)), sep='\n\n')
# #// print(env_samples)

# # print(*list(zip(a_states.tolist(),env_samples.tolist())), sep='\n\n')
# # exit()

# #_ Exclude irrelivant pheros based on `a_states`
# processed = []
# for a_state, env_sample in zip(a_states.tolist(), env_samples.tolist()):
#     if a_state == 0:
#         # Apply excludePheroB to every cell in the row
#         processed_row = [Cell.getPheroA(cell) for cell in env_sample]
#     else:
#         # Apply excludePheroA to every cell in the row
#         processed_row = [Cell.getPheroB(cell) for cell in env_sample]
#     processed.append(processed_row)

# # Convert to tensor
# env_samples = torch.tensor(processed, dtype=torch.int64)

# #_ Normalize input samples
# batch_norm_neighbours = torch.vmap(lambda x: torch.vmap(lambda y: y/torch.max(torch.max(x),torch.tensor(1)))(x))
# norm_env_samples = torch.tensor(batch_norm_neighbours(env_samples), dtype=torch.float64)

# #_ Bring bearings and samples together into one tensor of inputs
# print(bearings.shape, norm_env_samples.shape)

# inputs = torch.cat([bearings,norm_env_samples], dim = 1)

# print(*list(zip(inputs.tolist(),env_samples.tolist())), sep='\n\n')
# print(inputs.dtype)

#! Test Genotype <-> Weights + Biases
# layers = [1,3,3,1]
# Ws = torch.nn.ParameterList([
#     torch.nn.Parameter(torch.Tensor(
#         [[ 1, 2, 3]]
#     )),
#     torch.nn.Parameter(torch.Tensor(
#         [[ 4, 5, 6],
#          [ 7, 8, 9],
#          [10,11,12]]
#     )),               
#     torch.nn.Parameter(torch.Tensor(
#         [[13],
#          [14],
#          [15]]
#     ))
# ])
# bs = torch.nn.ParameterList([
#     torch.nn.Parameter(torch.Tensor([[101,102,103]])),
#     torch.nn.Parameter(torch.Tensor([[104,105,106]])),
#     torch.nn.Parameter(torch.Tensor([[107]]))
# ])

# a = AgentNN(layers)
# print(*list(a.named_parameters()), sep='\n', end='\n\n')
# print(*Ws, *bs, sep='\n', end='\n\n')

# g = AgentNN.get_genotype_from_wb(Ws,bs).tolist()
# print(g, end='\n\n')

# pWs, pbs = AgentNN.get_wb_from_genotype(g,layers)

# print(*pWs.parameters(), *pbs.parameters(), sep='\n')


#! Test getting optimizer and scheduler params
# agent = AgentNN([4,3,1])

# optim = torch.optim.Adam(agent.parameters())
# sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, min_lr=1e-8)

# print({"name":optim.__class__.__name__}|{k:v for k,v in optim.param_groups[0].items() if k != 'params'})
# print({"name":sched.__class__.__name__}|sched.state_dict())


#_ Generate random bearings
# random_states = 10

# bearings = torch.cat([ #? Shape: [random_states, 2]
#     (torch.rand(1) * 2*pi)
#         for i in range(random_states)
# ])
# print(bearings)

# b = torch.cat([ #? Shape: [random_states, 2]
#     (lambda b: torch.FloatTensor([np.sin(b),np.cos(b)]))(bearings[i])
#         .unsqueeze(0) for i in range(random_states)
# ])
# print(b)

# vector_to_angle = torch.vmap(lambda v: torch.arctan2(v[0],v[1]) % (2*pi))

# print(vector_to_angle(b))

# b = torch.tensor(0)

# c,s = torch.cos(b), -torch.sin(b)

# at = torch.arctan2(-s, c)

# print(b, s, c, at)

# a = Agent(layers=[1,2,1])

# # params = torch.nn.ParameterList([p for p in a.parameters()])

# args = {"name":"reducelronplateau",}

# fn = TestAgentNN._get_optimizer_func(params=a.parameters(), **args)

# print(fn)
# print()

# %%
sample = np.arange(9).reshape([3,3])
sample = sample.flatten()
neighbours, current_pos = np.concat([sample[:4],sample[5:]]), sample[4]

print(sample,neighbours,current_pos, sep='\n'*2, end='\n'*3)

delta_n = neighbours - current_pos

weights = np.where(
    delta_n < 0, 0.5, np.where(
        delta_n == 0, 1, np.where(
            delta_n > 0, 1 + delta_n, 0
        )
    )
)

t_probs = weights / np.sum(weights)

print(t_probs)

# %%
items = np.array([
    [1,1,0],
    [1,0,0],
    [0,0,0]
])
t_probs = np.arange(9).reshape([3,3])

mask = (
    np.where( # if, then
        items == Cell.item.FOOD, True,
    np.where( # elif, then
        items == Cell.item.NEST, True,
    False # else then
)))
if np.any(mask):
    t_probs = np.where(mask, 1, 0)

print(t_probs)
