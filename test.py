import numpy as np

from src.cell import Cell

# c = 0

# c = Cell.setState(c, 1)
# c = Cell.setPheroA(c, 5)
# c = Cell.setPheroB(c, 4)

# print(Cell.getState(c))
# print(Cell.getPheroA(c))
# print(Cell.getPheroB(c))
# print()

# c = Cell.setPheroA(c, 0x7FFFFFFF)

# print(Cell.getState(c))
# print(Cell.getPheroA(c))
# print(Cell.getPheroB(c))
# print()
# print(0x7FFFFFFF)

# xy_phero = 0x7FFFFFFF
# for _ in range(2):
#     blur = xy_phero / 9

#     diffusionDelta = 0.01
#     evaporationDelta = 0.01 * 0x7FFFFFFF

#     diff = (diffusionDelta * xy_phero) + ((1-diffusionDelta) * blur)
#     diff_evap = max(0, diff - evaporationDelta)

#     print(f"{round(diff_evap, 3):,}")

#     xy_phero = diff_evap


env = np.random.randint(0,2,(5,5))
pos = 4,4


print(env)
print()

# padded_env = np.pad(env, pad_width=1, mode='constant', constant_values=-1)

surr = np.pad(env, pad_width=1, mode='constant', constant_values=-1)[pos[0]:pos[0]+3, pos[1]:pos[1]+3]

print(surr)
print()

if any(np.array_equal([-1, -1, -1], edge) for edge in (surr[0,:], surr[2,:], surr[:,0], surr[:,2])):
    print('At edge!')

if (surr[0,:] == [-1,-1,-1]).all(): print('Top')
if (surr[2,:] == [-1,-1,-1]).all(): print('Bottom')
if (surr[:,0] == [-1,-1,-1]).all(): print('Left')
if (surr[:,2] == [-1,-1,-1]).all(): print('Right')