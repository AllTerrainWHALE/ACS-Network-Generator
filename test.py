import numpy as np

from math import pi, radians, degrees

from src.cell import Cell

#? Print statement to stop the first output being merged with shit in the terminal startup
#? IDK what the fuck goes on there
print()

#! Cell Class Testing
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
# env = np.full((3,3),0,dtype=np.int64)
# env[0,1] = Cell.setPheroA(env[0,1],0x7FFFFFFF)
# env[2,0] = Cell.setPheroA(env[2,0],0x7FFFFFFF)
# # poss = np.array([(1,2),(2,2),(0,1)])
# pos = np.array([4,6])
# bearing = 2*pi/4
# state = 0

# print(f"Environment:\n{env}", end='\n\n')

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

import numpy as np

def bearing_to_transformation(bearing):
    """
    Convert a bearing in radians to a (x, y) transformation.
    
    Args:
    - bearing (float): Angle in radians.
    
    Returns:
    - tuple: (x, y) transformation as integers in [-1, 0, 1].
    """
    # Compute x, y from the bearing
    x = np.cos(bearing)
    y = -np.sin(bearing)  # Negative sign to invert the vertical axis
    
    # Round to nearest integers in [-1, 0, 1]
    return round(x), round(y)

# Example bearings
bearings = [0, np.pi/2, np.pi, 3*np.pi/2, np.pi/4, 7*np.pi/4]
transformations = [bearing_to_transformation(b) for b in bearings]

# Output results
for bearing, (x, y) in zip(bearings, transformations):
    print(f"Bearing: {degrees(bearing):.2f} degrees -> Transformation: ({x}, {y})")

def transformation_to_index(x, y):
    """
    Convert a (x, y) transformation to a 1D index of a 3x3 grid.
    
    Args:
    - x (int): Horizontal transformation (-1, 0, 1).
    - y (int): Vertical transformation (-1, 0, 1).
    
    Returns:
    - int: 1D index of the 3x3 grid.
    """
    # Map (x, y) to grid coordinates
    grid_row = y + 1
    grid_col = x + 1
    
    # Compute 1D index
    return grid_row * 3 + grid_col

# Example transformations
transformations = [(1, 0), (0, -1), (-1, 0), (0, 1), (1, 1), (-1, -1), (1,-1), (-1,1)]
indices = [transformation_to_index(x, y) for x, y in transformations]

# Output results
for (x, y), idx in zip(transformations, indices):
    print(f"Transformation ({x}, {y}) -> Index: {idx}")

