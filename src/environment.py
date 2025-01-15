import numpy as np
import threading as th
import pygame as pg
import os

from numba import cuda
from time import time

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

        self.update_dt = [0]
        self.ups = 0

        self.thread_lock = th.Lock()
        
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
    #             if tx+dx < 0 or tx+dx > nx: continue

    #             for dy in range(-1, 2):
    #                 if ty+dy < 0 or ty+dy > ny: continue

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

    def _static_disperseAndEvaporate(grid, output_grid:list[int,int], dt:float=1):
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

            diffusionDelta = 0.0000001 * dt
            evaporationDelta = .001 * 0x7FFFFFFF * dt

            # Diffuse and evaporate pheromones
            diff_evap_a = max(0, ((diffusionDelta * xy_pheroA) + ((1 - diffusionDelta) * blur_a)) - evaporationDelta)
            diff_evap_b = max(0, ((diffusionDelta * xy_pheroB) + ((1 - diffusionDelta) * blur_b)) - evaporationDelta)


            # Apply effects
            output_grid[x, y] = ((int(diff_evap_a) & 0x7FFFFFFF) << 31) | (int(diff_evap_b) & 0x7FFFFFFF)
    
    def disperseAndEvaporate(self, dt:float=1):

        self.grid_device.copy_to_device(self.grid)

        # Launch the kernel on the GPU with the appropriate configuration
        threads_per_block = (32, 32)  # A 16x16 block of threads (256 threads per block)
        blocks_per_grid = (self.grid.shape[0] // threads_per_block[0], self.grid.shape[1] // threads_per_block[1])

        # Ensure there is at least 1 block in each dimension
        blocks_per_grid = (max(1, blocks_per_grid[0]), max(1, blocks_per_grid[1]))

        # Launch the GPU kernel with the specified block and grid size
        self._static_disperseAndEvaporate[blocks_per_grid, threads_per_block](self.grid_device, self.output_grid_device, dt)

        # Safely copy the new grid back to host
        with self.thread_lock:
            self.grid = self.output_grid_device.copy_to_host()

    def update(self, dt:float=1):
        start = time()

        self.disperseAndEvaporate(self.update_dt[-1])

        self.update_dt.append(time() - start)
        self.update_dt = self.update_dt[-100:]
        self.ups = len(self.update_dt) / sum(self.update_dt)
        # print(np.array([[Cell.getPheroA(cx) for cx in cy] for cy in self.grid]), end='\n\n')

    def get_grid_safely(self):
        # Safely access the grid in threading
        with self.thread_lock:
            return self.grid
        
    def get_ups_safely(self):
        with self.thread_lock:
            return self.ups
        





class Visualiser:
    def __init__(
        self, env:Environment,

        screen_res:tuple[int,int]=None,
        fullscreen:bool=False,
        fps:int=30,
        bg_color:tuple=(0,0,0), fg_color:tuple=(0,204,204)
    ):
        self.env = env

        # os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,0)
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        pg.init()
        pg.display.set_caption("Environment Visualiser")
        # pg.mouse.set_visible(False)

        if fullscreen:
            disp = pg.display.Info()
            screen_res = disp.current_w, disp.current_h

        elif screen_res == None:
            screen_res = self.env.grid.shape


        self.screen = pg.display.set_mode(screen_res)
        self.rect = self.screen.get_rect()

        self.clock = pg.time.Clock()
        self.dt = 0
        self.fps = fps
        
        self.font = pg.font.SysFont('impact', 13)

        self.bg = bg_color
        self.fg = fg_color
        self.screen.fill(self.bg)
        
        self.running = True
        self.playing = True

        # Dummy Agent
        self.posx, self.posy = 250,250

    def tick(self,dt):
        if self.playing:
            pass
            # self.posy = min(self.env.grid.shape[1], max(0, self.posy - (10*dt)))

            # self.env.grid[int(self.posx),int(self.posy)] = Cell.setPheroA(self.env.grid[int(self.posx),int(self.posy)], 0x7FFFFFFF)

    def render(self,dt):

        fps_txt = self.font.render(f'{round(self.clock.get_fps())} FPS', True, pg.Color('grey'))
        fps_txt_rect = fps_txt.get_rect(topleft=(0,0))

        ups_txt = self.font.render(f'{round(self.env.get_ups_safely())} UPS', True, pg.Color('grey'))
        ups_txt_rect = ups_txt.get_rect(topleft=(0,20))

        get_all_cell_data = np.vectorize(lambda g: Cell.getAll(g))
        # get_all_pheroB = np.vectorize(lambda g: Cell.getPheroB(g))

        states,pheroA,pheroB = get_all_cell_data(self.env.get_grid_safely())

        pheroA_norm = pheroA / 2147483647
        pheroB_norm = pheroB / 2147483647
        # pheroA_scaled = np.clip(pheroA_norm * 1, 0, 1)

        g = pheroA_norm[:,:,np.newaxis] * np.array((0,204,0))[np.newaxis,np.newaxis,:]
        g += pheroB_norm[:,:,np.newaxis] * np.array((204,0,0))[np.newaxis,np.newaxis,:]
        # g += states[states == 1] * np.array((255,255,255))[np.newaxis,np.newaxis,:]

        surf = pg.surfarray.make_surface(g)
        surf = pg.transform.scale_by(surf,(
            self.screen.get_width() / self.env.grid.shape[0],
            self.screen.get_height() / self.env.grid.shape[1]
        ))

        self.screen.blit(surf, (0,0))
        self.screen.blit(fps_txt,fps_txt_rect)
        self.screen.blit(ups_txt,ups_txt_rect)
        
        pg.display.update()

    def main(self):
        while self.running:
            dt = self.clock.tick(self.fps) / 1000

            events = pg.event.get()
            for event in events:
                if event.type == pg.QUIT:
                    pg.quit()
                    return
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:
                        self.playing = not self.playing

            self.tick(dt)
            self.render(dt)