import numpy as np

from src.environment import Environment
from src.cell import Cell
from time import time

if __name__ == '__main__':
    env = Environment((1920,1080), 1000)
    env.grid[4,4] = Cell.setPheroA(env.grid[1,1], 0x7FFFFFFF)

    print('INITIAL GRID')
    print(np.array([[Cell.getPheroA(cx) for cx in cy] for cy in env.grid]), end='\n\n')
    
    durations = []
    for _ in range(100):
        start = time()
        env.update()
        durations.append(time() - start)

    print(*durations, sep='\n')
    print()
    print('Average:', sum(durations) / len(durations))