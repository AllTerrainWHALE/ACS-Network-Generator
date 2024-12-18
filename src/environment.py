import numpy as np

from numba import jit, cuda, prange
from src.cell import Cell

class Environment:

    def __init__(self, resolution:tuple[int,int], population:int=0):

        self.grid = np.zeros((resolution), dtype=np.int64)
        # self.grid = np.array([[0 for x in range(resolution[0])] for y in range(resolution[1])])
        # np.array([[cx.setState(3) for cx in cy] for cy in self.grid])
        # print(np.array([[cx.getState() for cx in cy] for cy in self.grid]))
        
    @staticmethod
    @cuda.jit
    def _static_disperseAndEvaporate(grid, output_grid, dt:float=1):
        #assert len(np.shape(grid)) == 2, "Grid is not 2-Dimensional"

        # Thread indices
        x, y = cuda.grid(2)

        #return grid[1:-1,1:-1]

        # for y in prange(1, len(grid)-1):
        #     for x in prange(1, len(grid[y])-1):
        if x >= 1 and x < grid.shape[0] - 1 and y >= 1 and y < grid.shape[1] - 1:
            sum_a = 0
            sum_b = 0
            for x1 in range(-1,2):
                for y1 in range(-1,2):
                    sum_a += (grid[x + x1, y + y1] >> 31) & 0x7FFFFFFF #Cell.getPheroA(grid[x+x1][y+y1])
                    sum_b += grid[x + x1, y + y1] & 0x7FFFFFFF #Cell.getPheroB(grid[x+x1][y+y1])
            blur_a = sum_a / 9
            blur_b = sum_b / 9

            diffusionDelta = 1 * dt

            # Diffuse Pheromones A & B
            diff_a = (diffusionDelta * ((grid[x, y] >> 31) & 0x7FFFFFFF)) + ((1-diffusionDelta) * blur_a)
            diff_b = (diffusionDelta * (grid[x, y] & 0x7FFFFFFF)) + ((1-diffusionDelta) * blur_b)

            # Evaporate Pheromones A & B
            diff_evap_a = max(0, diff_a - (0.01 * dt))
            diff_evap_b = max(0, diff_b - (0.01 * dt))

            # Apply effects
            output_grid[x, y] = (output_grid[x, y] & ~0x7FFFFFFF) | ((int(diff_evap_a) & 0x7FFFFFFF) << 31)
            # Set PheroB (bits 0 to 30)
            output_grid[x, y] = (output_grid[x, y] & ~0x7FFFFFFF) | (int(diff_evap_b) & 0x7FFFFFFF)

        # return output_grid[1:-1,1:-1]
    
    def disperseAndEvaporate(self, dt:float=1):
        # Add padding around grid to avoid null references
        # padded_grid = np.array([[0 for x in range(self.grid.shape[1]+2)] for y in range(self.grid.shape[0]+2)])
        padded_grid = np.zeros(np.add(self.grid.shape,(2,2)), dtype=np.int64)
        padded_grid[1:-1,1:-1] = self.grid

        # Prepare devices
        padded_grid_device = cuda.to_device(padded_grid)
        output_grid_device = cuda.device_array_like(padded_grid)

        # Launch the kernel on the GPU with the appropriate configuration
        threads_per_block = (16, 16)  # A 16x16 block of threads (256 threads per block)
        blocks_per_grid = (padded_grid.shape[0] // threads_per_block[0], padded_grid.shape[1] // threads_per_block[1])

        # Ensure there is at least 1 block in each dimension
        blocks_per_grid = (max(1, blocks_per_grid[0]), max(1, blocks_per_grid[1]))

        # Launch the GPU kernel with the specified block and grid size
        self._static_disperseAndEvaporate[blocks_per_grid, threads_per_block](padded_grid_device, output_grid_device, dt)

        # Copy the new grid back to host
        self.grid = output_grid_device.copy_to_host()[1:-1,1:-1]

    def update(self, dt:float=1):
        self.disperseAndEvaporate(dt)