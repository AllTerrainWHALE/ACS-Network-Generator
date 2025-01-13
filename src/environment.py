import numpy as np

from numba import jit, cuda, prange
from src.cell import Cell

class Environment:

    def __init__(self, resolution:tuple[int,int], population:int=0):

        self.grid = np.zeros((resolution), dtype=np.int64)
        # self.grid = np.array([[0 for x in range(resolution[0])] for y in range(resolution[1])])
        # np.array([[cx.setState(3) for cx in cy] for cy in self.grid])
        # print(np.array([[cx.getState() for cx in cy] for cy in self.grid]))

        # Allocate device memory for grid and output grid once
        self.grid_device = cuda.to_device(self.grid)
        self.output_grid_device = cuda.device_array_like(self.grid)
        
    @staticmethod
    @cuda.jit
    # def _static_disperseAndEvaporate(grid, output_grid, dt: float = 1):
    #     # Thread indices
    #     x, y = cuda.grid(2)

    #     # Grid size
    #     nx, ny = grid.shape

    #     # Shared memory for a block (assumes blockDim.x and blockDim.y <= 16)
    #     shared_grid = cuda.shared.array((32 + 2, 32 + 2), dtype=np.int32)

    #     # Local thread indices in the block
    #     tx = cuda.threadIdx.x + 1
    #     ty = cuda.threadIdx.y + 1

    #     # Load data into shared memory with a halo
    #     if x < nx and y < ny:
    #         shared_grid[tx, ty] = grid[x, y]
    #         # Load the halo regions
    #         if cuda.threadIdx.x == 0 and x > 0:
    #             shared_grid[tx - 1, ty] = grid[x - 1, y]
    #         if cuda.threadIdx.x == cuda.blockDim.x - 1 and x < nx - 1:
    #             shared_grid[tx + 1, ty] = grid[x + 1, y]
    #         if cuda.threadIdx.y == 0 and y > 0:
    #             shared_grid[tx, ty - 1] = grid[x, y - 1]
    #         if cuda.threadIdx.y == cuda.blockDim.y - 1 and y < ny - 1:
    #             shared_grid[tx, ty + 1] = grid[x, y + 1]
    #     cuda.syncthreads()

    #     if x < nx and y < ny:
    #         # Decode pheromones
    #         xy_pheroA = (shared_grid[tx, ty] >> 31) & 0x7FFFFFFF
    #         xy_pheroB = shared_grid[tx, ty] & 0x7FFFFFFF

    #         # Blur calculation (sum of neighbors)
    #         sum_a = 0
    #         sum_b = 0
    #         for dx in range(-1, 2):
    #             for dy in range(-1, 2):
    #                 neighbor_val = shared_grid[tx + dx, ty + dy]
    #                 sum_a += (neighbor_val >> 31) & 0x7FFFFFFF
    #                 sum_b += neighbor_val & 0x7FFFFFFF

    #         blur_a = sum_a / 9
    #         blur_b = sum_b / 9

    #         diffusionDelta = 0.5 * dt

    #         # Diffuse and evaporate pheromones
    #         diff_evap_a = max(0, (diffusionDelta * xy_pheroA + (1 - diffusionDelta) * blur_a) - (0.01 * dt))
    #         diff_evap_b = max(0, (diffusionDelta * xy_pheroB + (1 - diffusionDelta) * blur_b) - (0.01 * dt))

    #         # Write to output grid
    #         output_grid[x, y] = ((int(diff_evap_a) & 0x7FFFFFFF) << 31) | (int(diff_evap_b) & 0x7FFFFFFF)

    def _static_disperseAndEvaporate(grid, output_grid, dt:float=1):
        # assert len(np.shape(grid)) == 2, "Grid is not 2-Dimensional"

        # Thread indices
        x, y = cuda.grid(2)

        if x >= 0 and x < grid.shape[0] and y >= 0 and y < grid.shape[1]:
            xy_pheroA = (grid[x, y] >> 31) & 0x7FFFFFFF
            xy_pheroB = grid[x, y] & 0x7FFFFFFF

            sum_a = 0
            sum_b = 0
            for x1 in range(-1,2):
                if x+x1 < 0 or x+x1 > grid.shape[0]: continue
                for y1 in range(-1,2):
                    if y+y1 < 0 or y+y1 > grid.shape[1]: continue

                    sum_a += (grid[x + x1, y + y1] >> 31) & 0x7FFFFFFF # Get PheroA value
                    sum_b += grid[x + x1, y + y1] & 0x7FFFFFFF # Get PheroB value
            blur_a = sum_a / 9
            blur_b = sum_b / 9

            diffusionDelta = 0.5 * dt

            # # Diffuse & Evaporate Pheromones A
            # if blur_a != -1:
            #     diff_a = (diffusionDelta * xy_pheroA) + ((1-diffusionDelta) * blur_a) # diffuse
            #     diff_evap_a = max(0, diff_a - (0.01 * dt)) # evaporate
            # else:
            #     diff_evap_a = 0

            # # Diffuse & Evaporate Pheromones B
            # if blur_a != -1:
            #     diff_b = (diffusionDelta * xy_pheroB) + ((1-diffusionDelta) * blur_b) # diffuse
            #     diff_evap_b = max(0, diff_b - (0.01 * dt)) # evaporate
            # else:
            #     diff_evap_b = 0

            # Diffuse and evaporate pheromones
            diff_evap_a = max(0, (diffusionDelta * xy_pheroA + (1 - diffusionDelta) * blur_a) - (0.01 * dt))
            diff_evap_b = max(0, (diffusionDelta * xy_pheroB + (1 - diffusionDelta) * blur_b) - (0.01 * dt))


            # Apply effects
            output_grid[x, y] = ((int(diff_evap_a) & 0x7FFFFFFF) << 31) | (int(diff_evap_b) & 0x7FFFFFFF)
    
    def disperseAndEvaporate(self, dt:float=1):
        # Add padding around grid to avoid null references
        # padded_grid = np.array([[0 for x in range(self.grid.shape[1]+2)] for y in range(self.grid.shape[0]+2)])
        # padded_grid = np.zeros(np.add(self.grid.shape,(2,2)), dtype=np.int64)
        # padded_grid[1:-1,1:-1] = self.grid

        # # Prepare devices
        # padded_grid_device = cuda.to_device(padded_grid)
        # output_grid_device = cuda.device_array_like(padded_grid)

        self.grid_device.copy_to_device(self.grid)

        # Launch the kernel on the GPU with the appropriate configuration
        threads_per_block = (32, 32)  # A 16x16 block of threads (256 threads per block)
        blocks_per_grid = (self.grid.shape[0] // threads_per_block[0], self.grid.shape[1] // threads_per_block[1])

        # Ensure there is at least 1 block in each dimension
        blocks_per_grid = (max(1, blocks_per_grid[0]), max(1, blocks_per_grid[1]))

        # Launch the GPU kernel with the specified block and grid size
        self._static_disperseAndEvaporate[blocks_per_grid, threads_per_block](self.grid_device, self.output_grid_device, dt)

        # Copy the new grid back to host
        self.grid = self.output_grid_device.copy_to_host()

    def update(self, dt:float=1):
        self.disperseAndEvaporate(dt)
        # print(np.array([[Cell.getPheroA(cx) for cx in cy] for cy in self.grid]), end='\n\n')