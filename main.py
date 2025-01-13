import numpy as np

from src.environment import Environment
from src.cell import Cell
from time import time

if __name__ == '__main__':
    env = Environment((1920,1080), 1000)
    # env = Environment((5,5), 1000)

    env.grid[2,2] = Cell.setPheroA(env.grid[2,2], 50000)
    # env.grid[2,2] = Cell.setPheroB(env.grid[2,2], 1000)

    print('INITIAL GRID')
    print(np.array([[Cell.getPheroA(cx) for cx in cy] for cy in env.grid]), end='\n\n')
    # print(np.array([[Cell.getPheroB(cx) for cx in cy] for cy in env.grid]), end='\n\n')
    
    durations = []
    for _ in range(1000):
        start = time()
        env.update()
        # print(np.array([[Cell.getPheroA(cx) for cx in cy] for cy in env.grid]), end='\n\n')
        # print(np.array([[Cell.getPheroB(cx) for cx in cy] for cy in env.grid]), end='\n\n')
        durations.append(time() - start)

    print(*durations, sep='\n')
    print()

    average_dur = sum(durations[1:]) / len(durations[1:])
    ups = 1 / average_dur

    print('Average:', average_dur)
    print('UPS:\t', ups)