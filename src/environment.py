import numpy as np
import threading as th
import pygame as pg
import os

from numba import cuda
from time import time
from datetime import timedelta
from math import ceil, sqrt

# from src.colony import Colony
# from src.agent import Agent
from src.cell import Cell

class Environment:

    def __init__(self, resolution:tuple[int,int], food_sources:bool=True):
        assert isinstance(resolution, (list, tuple, np.ndarray)), f"`resolution` must be a array, list or tuple"
        assert len(resolution) == 2, f"Environment can only be a 2-Dimensional shape, not {len(resolution)}-Dimensional"
        #// assert isinstance(colonies, (list, tuple, np.ndarray)), f"> `colonies` must be a list or tuple"
        #// assert len(np.shape(colonies)) == 1, f"> `colonies` list must be 1-Dimensional"

        #_ Define environment grid
        self.grid = np.zeros((resolution), dtype=np.uint64)
        self.p_grid = np.pad(self.grid, pad_width=1, mode='constant', constant_values=Cell.setState(0, 3))

        #_ Place food sources
        if food_sources:
            # self.placeFoodDeposit(np.random.randint(150,350, 2))
            self.placeFoodDeposit([200,250])
            self.placeObstructionSquare([210,215], [275,225])

        #_ Allocate device memory for grid and output grid once
        self.grid_device = cuda.to_device(self.grid)
        self.output_grid_device = cuda.device_array_like(self.grid)

        self.colonies = []

        #// self.agents = np.array([Agent((250,250), state=1) for a in range(population)])
        #// (np.random.randint(resolution[0]), np.random.randint(resolution[1]))

        self.paused = False
        self.runtime_start = None
        self.update_dt = [0]
        self.ups = 0

        self.thread_lock = th.Lock()

        self.disperseAndEvaporate()
        
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
            xy_pheroA = (grid[x, y] >> 30) & 0x3FFFFFFF
            xy_pheroB = grid[x, y] & 0x3FFFFFFF
            isWall = (grid[x, y] >> 60) & 0b11 == 3

            if not isWall:
                sum_a = 0
                sum_b = 0
                for x1 in range(-1,2):
                    if x+x1 < 0 or x+x1 >= grid.shape[0]: continue
                    for y1 in range(-1,2):
                        if y+y1 < 0 or y+y1 >= grid.shape[1]: continue

                        sum_a += (grid[x + x1, y + y1] >> 30) & 0x3FFFFFFF # Get PheroA value
                        sum_b += grid[x + x1, y + y1] & 0x3FFFFFFF # Get PheroB value
                blur_a = sum_a / 9
                blur_b = sum_b / 9

                diffusionDelta = 0.7 * dt
                evaporationDelta = 0.01 * dt

                # Diffuse and evaporate pheromones
                diff_evap_a = max(0, min((xy_pheroA + diffusionDelta * (blur_a - xy_pheroA)) * (1 - evaporationDelta), 0x3FFFFFFF))
                diff_evap_b = max(0, min((xy_pheroB + diffusionDelta * (blur_b - xy_pheroB)) * (1 - evaporationDelta), 0x3FFFFFFF))
            
            else:
                diff_evap_a = diff_evap_b = 0
            #// diff_evap_a = max(0, min(((diffusionDelta * blur_a) + ((1 - diffusionDelta) * xy_pheroA)) - evaporationDelta, 0x7FFFFFFF))
            #// diff_evap_b = max(0, min(((diffusionDelta * blur_b) + ((1 - diffusionDelta) * xy_pheroB)) - evaporationDelta, 0x7FFFFFFF))

            # Apply effects
            # output_grid[x, y] = ((int(diff_evap_a) & 0x7FFFFFFF) << 31) | (int(diff_evap_b) & 0x7FFFFFFF)
            output_grid[x,y] = (grid[x,y] & ~((0x3FFFFFFF << 30) | 0x3FFFFFFF)) | ((int(diff_evap_a) & 0x3FFFFFFF) << 30) | (int(diff_evap_b) & 0x3FFFFFFF)

    
    def disperseAndEvaporate(self, dt:float=1):

        self.grid_device.copy_to_device(self.grid)

        # Launch the kernel on the GPU with the appropriate configuration
        threads_per_block = (32, 32)  # A 16x16 block of threads (256 threads per block)
        blocks_per_grid = (ceil(self.grid.shape[0] / threads_per_block[0]), ceil(self.grid.shape[1] / threads_per_block[1]))

        # Ensure there is at least 1 block in each dimension
        blocks_per_grid = (max(1, blocks_per_grid[0]), max(1, blocks_per_grid[1]))

        # Launch the GPU kernel with the specified block and grid size
        self._static_disperseAndEvaporate[blocks_per_grid, threads_per_block](self.grid_device, self.output_grid_device, dt)

        # Safely copy the new grid back to host
        with self.thread_lock:
            self.grid = self.output_grid_device.copy_to_host()
            # print(self.grid[250,250])
            self.p_grid[1:-1,1:-1] = self.grid

    def update(self, dt:float=1):
        if self.paused: return self.update_dt[-1]

        if self.runtime_start == None:
            self.runtime_start = time()

        start = time()
        dt = self.update_dt[-1]



        ###! TESTING !###
        # self.grid[275,(225,250)] = Cell.setPheroB(0,0x3FFFFFFF)
        # self.grid[225,(225,275)] = Cell.setPheroB(0,0x3FFFFFFF)
        # self.grid[50:230, 230] = Cell.setPheroB(0,0x3FFFFFFF)

        if time() - self.runtime_start >= 60:
            self.grid[245:255, 215:225] = np.vectorize(lambda c: Cell.setState(c, Cell.item.NONE))(self.grid[245:255, 215:225])


        for colony in self.colonies:
            colony.update(dt)

        self.disperseAndEvaporate(dt)

        self.update_dt.append(time() - start)
        self.update_dt = self.update_dt[-100:]
        self.ups = len(self.update_dt) / sum(self.update_dt)

        return self.update_dt[-1]

    def get_grid_safely(self):
        # Safely access the grid in threading
        with self.thread_lock:
            return self.grid
        
    def get_ups_safely(self):
        with self.thread_lock:
            return self.ups
        
    def placeFoodDeposit(self, pos:"list[int]|tuple[int]", radius:int=5):
        r,c = self.grid.shape
        y,x = np.ogrid[:r,:c]

        distance = (x - pos[0])**2 + (y - pos[1])**2 # distance^2 from circle center
        mask = distance <= radius**2

        self.grid[mask] = Cell.setState(0, int(Cell.item.FOOD))

    def placeObstructionSquare(self, pos1, pos2):
        self.grid[pos1[0]:pos2[0], pos1[1]:pos2[1]] = Cell.setState(0, int(Cell.item.WALL))
        
    

    @staticmethod
    def sample_state(env:"Environment"=None, empty_ratio:float=0.3, items:bool=True):
        if env == None:
            env = Environment([3,3], food_sources=False)
        else:
            env.grid = np.zeros_like(env.grid, dtype=np.uint64)

        assert env.grid.shape == (3,3), f"Sample environment must have a resolution of (3, 3), not {env.grid.shape}"

        index = np.random.randint(0,3), np.random.randint(0,3)

        while items:
            item_list = np.random.choice([0, 1, 2, 3], size=(3, 3), p=[0.88, 0.01, 0.01, 0.1])
            if np.any(np.delete(item_list,4) != Cell.item.WALL):
                env.grid = item_list.astype(np.uint64) << 60
                break
        
        if np.random.rand() < empty_ratio:  # Force x% zero-pheromone samples
            env.grid[index] = Cell.setPheroA(env.grid[index], 0)
            env.grid[index] = Cell.setPheroB(env.grid[index], 0)
        else:
            env.grid[index] = Cell.setPheroA(env.grid[index], np.random.randint(0x3FFFFFFF))
            env.grid[index] = Cell.setPheroB(env.grid[index], np.random.randint(0x3FFFFFFF))

        for _ in range(np.random.randint(10)):
            env.disperseAndEvaporate()

        return env.grid



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

        self.bg = np.array(bg_color)
        self.fg = np.array(fg_color)
        self.screen.fill(self.bg)

        # self.runtime_start = time()
        
        self.running = True
        self.playing = True

        # Dummy Agent
        self.posx, self.posy = 250,250

    def tick(self,dt):
        if self.playing:
            pass

    def render(self,dt):

        fps_txt = self.font.render(f'{round(self.clock.get_fps())} FPS', True, pg.Color('grey'))
        fps_txt_rect = fps_txt.get_rect(topleft=(0,0))

        ups_txt = self.font.render(f'{round(self.env.get_ups_safely())} UPS', True, pg.Color('grey'))
        ups_txt_rect = ups_txt.get_rect(topleft=(0,20))

        rt_txt = self.font.render(f'{timedelta(seconds=round(time() - self.env.runtime_start))}', True, pg.Color('grey'))
        rt_txt_rect = rt_txt.get_rect(topright=self.rect.topright)

        ###_ Render Pheromones and Items _###
        get_all_cell_data = np.vectorize(lambda g: Cell.getAll(g))

        states,pheroA,pheroB = get_all_cell_data(self.env.get_grid_safely())

        #_ Normalize pheros
        pheroA_norm = (pheroA / 0x3FFFFFFF)
        pheroB_norm = (pheroB / 0x3FFFFFFF)
        
        #_ Scaling pheros
        gamma = 0.2
        pheroA_scale = np.power(pheroA_norm, gamma)
        pheroB_scale = np.power(pheroB_norm, gamma)

        #// scale_fact = 20
        #// pheroA_scale = ((scale_fact+1)*pheroA_norm) / ((pheroA_norm*scale_fact)+1)
        #// pheroB_scale = ((scale_fact+1)*pheroB_norm) / ((pheroB_norm*scale_fact)+1)

        #_ Boundarize pheros
        pheroA_clip = np.clip(pheroA_scale, 0, 1)
        pheroB_clip = np.clip(pheroB_scale, 0, 1)

        #_ Define phero colours
        pheroA_colour = (0,204,0)
        pheroB_colour = (0,0,204)

        #_ Add pheromones to canvas
        # amplifier = 0

        g = np.ones_like(pheroA_clip)[:,:,np.newaxis] * self.bg

        g += pheroA_clip[:,:,np.newaxis] * (np.array(pheroA_colour)[np.newaxis,np.newaxis,:] - g) # + amplifier * (
            # ((np.where(pheroA_clip >= 0.05, 1-pheroA_clip, 0)[:,:,np.newaxis]) * np.array(pheroA_colour)[np.newaxis,np.newaxis,:]*.5) +
            # ((np.where((pheroA_clip > 0.0) & (pheroA_clip < 0.05), 1-pheroA_clip, 0)[:,:,np.newaxis]) * np.array(pheroA_colour)[np.newaxis,np.newaxis,:]*.25)
        # )
        g += pheroB_clip[:,:,np.newaxis] * (np.array(pheroB_colour)[np.newaxis,np.newaxis,:] - g) # + amplifier * (
            # ((np.where(pheroB_clip >= 0.05, 1-pheroB_clip, 0)[:,:,np.newaxis]) * np.array(pheroB_colour)[np.newaxis,np.newaxis,:]*.5) +
            # ((np.where((pheroB_clip > 0.0) & (pheroB_clip < 0.05), 1-pheroB_clip, 0)[:,:,np.newaxis]) * np.array(pheroB_colour)[np.newaxis,np.newaxis,:]*.25)
        # )
        
        #_ Add environment items to canvas
        g[states == 1] = np.array((204,204,0))[np.newaxis,np.newaxis,:]
        g[states == 2] = np.array((153, 76,0))[np.newaxis,np.newaxis,:]
        g[states == 3] = np.array((102, 51,0))[np.newaxis,np.newaxis,:]

        ###_ Render Agents _###
        # ant_coords = np.array([a.get_pos() for a in self.env.colonies[0].agents])
        # g[tuple(ant_coords.T)] = (255,255,255)

        for colony in self.env.colonies:
            for agent in colony.agents:
                g[tuple(agent.get_pos())] = np.abs(self.bg - ((153,255,153) if agent.state == 1 else (153,153,255)))

        ###_ Blit canvas and add info HUD _###
        surf = pg.surfarray.make_surface(g)
        surf = pg.transform.scale_by(surf,(
            self.screen.get_width() / self.env.grid.shape[0],
            self.screen.get_height() / self.env.grid.shape[1]
        ))

        self.screen.blit(surf, (0,0))
        self.screen.blit(fps_txt,fps_txt_rect)
        self.screen.blit(ups_txt,ups_txt_rect)
        self.screen.blit(rt_txt,rt_txt_rect)
        
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
                        self.env.paused = not self.env.paused

            # self.tick(dt)
            self.render(dt)