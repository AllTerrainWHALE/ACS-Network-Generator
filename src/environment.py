import numpy as np
import threading as th
import pygame as pg
import os

from numba import cuda
from time import time
from datetime import timedelta
from math import ceil, sqrt

from src.cell import Cell

#_ Define kernal constants
MAX_PHERO = Cell.MAX_PHERO
MAX_ITEM = Cell.MAX_ITEM
MASK_STEP = Cell.MASK_STEP

DIFFUSION_RATE = 0.7
EVAPORATION_RATE = 0.1


class Environment:

    def __init__(
            self, resolution:tuple[int,int],

            food_sources:bool=True, food_source_positions:list[list[int,int]]=[],
            timestep:float=0.01,
            diffusion_rate:float=DIFFUSION_RATE, evaporation_rate:float=EVAPORATION_RATE
        ):
        assert isinstance(resolution, (list, tuple, np.ndarray)), f"`resolution` must be a array, list or tuple"
        assert len(resolution) == 2, f"Environment can only be a 2-Dimensional shape, not {len(resolution)}-Dimensional"

        #_ Define environment grid
        self.grid = np.zeros((resolution), dtype=np.uint64)
        self.p_grid = np.pad(self.grid, pad_width=1, mode='constant', constant_values=Cell.setItem(0, Cell.item.WALL))

        #_ Place food sources
        if food_sources:
            for pos in food_source_positions:
                self.placeFoodDeposit(pos) # Prefab node locations are given in (x,y) format, but the grid is (y,x) format

        #_ Allocate device memory for grid and output grid once
        self.grid_device = cuda.to_device(self.grid)
        self.output_grid_device = cuda.device_array_like(self.grid)

        self.colony = None

        #_ Additional attributes for controling runtime
        self.paused = False
        self.runtime_start = None
        self.internal_time = 0
        self.timestep = timestep
        self.update_dt = [0]
        self.ups = 0

        self.thread_lock = th.Lock()

        #_ Run the kernel once to load the device memory
        # This is to ensure that the kernel is compiled and ready to use
        self.disperseAndEvaporate()
        
    @staticmethod
    @cuda.jit
    def _static_disperseAndEvaporate(grid, output_grid:list[int,int], dt:float=1):
        # Thread indices
        x, y = cuda.grid(2)

        if x >= 0 and x < grid.shape[0] and y >= 0 and y < grid.shape[1]:
            xy_pheroA = (grid[x, y] >> MASK_STEP) & MAX_PHERO
            xy_pheroB = grid[x, y] & MAX_PHERO
            isWall = (grid[x, y] >> (MASK_STEP*2)) & MAX_ITEM == 3

            if not isWall: # Caluclate average pheromone values of A and B in the 3x3 grid
                sum_a = 0
                sum_b = 0
                for x1 in range(-1,2):
                    if x+x1 < 0 or x+x1 >= grid.shape[0]: continue
                    for y1 in range(-1,2):
                        if y+y1 < 0 or y+y1 >= grid.shape[1]: continue

                        sum_a += (grid[x + x1, y + y1] >> MASK_STEP) & MAX_PHERO # Get PheroA value
                        sum_b += grid[x + x1, y + y1] & MAX_PHERO # Get PheroB value
                blur_a = sum_a / 9
                blur_b = sum_b / 9

                # Calculate diffusion and evaporation factors in reference to the timestep
                diffusionFactor = DIFFUSION_RATE * dt
                evaporationFactor = EVAPORATION_RATE * dt

                # Diffuse and evaporate pheromones
                diff_evap_a = max(0, min((xy_pheroA + diffusionFactor * (blur_a - xy_pheroA)) * (1 - evaporationFactor), MAX_PHERO))
                diff_evap_b = max(0, min((xy_pheroB + diffusionFactor * (blur_b - xy_pheroB)) * (1 - evaporationFactor), MAX_PHERO))
            
            else: # Pheromones should not exist in walls
                diff_evap_a = diff_evap_b = 0

            # Apply updated pheromone values to the output grid
            output_grid[x,y] = (grid[x,y] & ~((MAX_PHERO << MASK_STEP) | MAX_PHERO)) | ((int(diff_evap_a) & MAX_PHERO) << MASK_STEP) | (int(diff_evap_b) & MAX_PHERO)

    
    def disperseAndEvaporate(self, dt:float=1):

        self.grid_device.copy_to_device(self.grid)

        # Launch the kernel on the GPU with the appropriate configuration
        threads_per_block = (16, 16)  # A 16x16 block of threads (256 threads per block)
        blocks_per_grid = (ceil(self.grid.shape[0] / threads_per_block[0]), ceil(self.grid.shape[1] / threads_per_block[1]))

        # Ensure there is at least 1 block in each dimension
        blocks_per_grid = (max(1, blocks_per_grid[0]), max(1, blocks_per_grid[1]))

        # Launch the GPU kernel with the specified block and grid size
        self._static_disperseAndEvaporate[blocks_per_grid, threads_per_block](self.grid_device, self.output_grid_device, dt)

        # Safely copy the new grid back to host
        with self.thread_lock:
            self.grid = self.output_grid_device.copy_to_host()
            self.p_grid[1:-1,1:-1] = self.grid

    def update(self, dt:float=1):
        #_ Don't do anything if paused
        if self.paused: return self.update_dt[-1]

        #_ Start the runtime clock
        if self.runtime_start == None:
            self.runtime_start = time()

        #_ Update runtime attributes
        start = time()
        dt = self.timestep
        self.internal_time += dt

        #_ Update the colony
        self.colony.update(dt)

        #_ Disperse and evaporate pheromones
        self.disperseAndEvaporate(dt)

        #_ Calculate and return the time taken for the update cycle
        self.update_dt.append(time() - start)
        self.update_dt = self.update_dt[-100:]
        self.ups = len(self.update_dt) / sum(self.update_dt) # Updates per second

        return self.update_dt[-1]

    def get_grid_safely(self):
        """
        Safely return the grid from the environment. This is to ensure that the grid is not modified while it is being used.
        """
        with self.thread_lock:
            return self.grid
        
    def get_ups_safely(self):
        """
        Safely return the updates per second from the environment. This is to ensure that the updates per second is not modified while it is being used.
        """
        with self.thread_lock:
            return self.ups
        
    def placeFoodDeposit(self, pos:"list[int]|tuple[int]", radius:int=5):
        """
        Place a food deposit in the environment at the given position with the given radius.
        """
        r,c = self.grid.shape
        x,y = np.ogrid[:r,:c]

        distance = (x - pos[0])**2 + (y - pos[1])**2 # distance^2 from circle center
        mask = distance <= radius**2

        self.grid[mask] = Cell.setItem(0, Cell.item.FOOD)

    def placeObstructionSquare(self, pos1:tuple[int,int], pos2:tuple[int,int]):
        """
        Place a square obstruction in the environment at the given position with the given radius.
        
        :param [int, int] pos1: The top left (x,y) position of the square.
        :param [int, int] pos2: The bottom right (x,y) position of the square.
        """
        self.grid[pos1[0]:pos2[0], pos1[1]:pos2[1]] = Cell.setItem(0, Cell.item.WALL)
        
    

    @staticmethod
    def sample_state(env:"Environment"=None, empty_ratio:float=0.3, items:bool=True) -> np.ndarray:
        """
        Generate a random sample state of the environment. The sample state is a 3x3 grid with random pheromone values and items.
        
        :param Environment env: The environment to generate the sample state for. If None, a new environment will be created.
        :param float empty_ratio: The ratio of empty cells in the sample state. The default is 0.3.
        :param bool items: Whether to include items in the sample state. The default is True.

        :return np.ndarray: The sample state as a 3x3 grid.
        """
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
            env.grid[index] = Cell.setPheroA(env.grid[index], np.random.randint(Cell.MAX_PHERO))
            env.grid[index] = Cell.setPheroB(env.grid[index], np.random.randint(Cell.MAX_PHERO))

        for _ in range(np.random.randint(10)):
            env.disperseAndEvaporate()

        return env.grid



class Visualiser:
    def __init__(
        self, env:Environment,

        screen_res:tuple[int,int]=None,
        fullscreen:bool=False,
        fps:int=30,
        bg_color:tuple=(255,255,255), fg_color:tuple=(0,204,204)
    ):
        self.env = env

        #// os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,0)
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        pg.init()
        pg.display.set_caption("Environment Visualiser")

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
        """
        Render the environment and the agents onto the canvas.
        """

        fps_txt = self.font.render(f'{round(self.clock.get_fps())} FPS', True, pg.Color('grey'))
        fps_txt_rect = fps_txt.get_rect(topleft=(0,0))

        ups_txt = self.font.render(f'{round(self.env.get_ups_safely())} UPS', True, pg.Color('grey'))
        ups_txt_rect = ups_txt.get_rect(topleft=(0,20))

        # rt_txt = self.font.render(f'{timedelta(seconds=round(time() - self.env.runtime_start))}', True, pg.Color('grey'))
        # rt_txt_rect = rt_txt.get_rect(topright=self.rect.topright)

        it_txt = self.font.render(f'{round(self.env.internal_time,3)}s', True, pg.Color('grey'))
        it_txt_rect = it_txt.get_rect(topright=self.rect.topright)

        dt_txt = self.font.render(f'{round(np.mean(self.env.update_dt),3)} spu', True, pg.Color('grey'))
        dt_txt_rect = dt_txt.get_rect(topright=self.rect.topright+pg.Vector2(0,20))

        ###_ Render Pheromones and Items _###
        get_all_cell_data = np.vectorize(lambda g: Cell.getAll(g))

        states,pheroA,pheroB = get_all_cell_data(self.env.get_grid_safely())

        #_ Normalize pheros
        pheroA_norm = (pheroA / Cell.MAX_PHERO)
        pheroB_norm = (pheroB / Cell.MAX_PHERO)
        
        #_ Scaling pheros
        gamma = .1
        pheroA_scale = np.power(pheroA_norm, gamma)
        pheroB_scale = np.power(pheroB_norm, gamma)

        #_ Boundarize pheros
        pheroA_clip = np.clip(pheroA_scale, 0, 1)
        pheroB_clip = np.clip(pheroB_scale, 0, 1)

        #_ Define phero colours
        pheroA_colour = (0,204,0)
        pheroB_colour = (0,0,204)

        #_ Add pheromones to canvas
        amplifier = 0

        g = np.ones_like(pheroA_clip)[:,:,np.newaxis] * self.bg

        g += pheroA_clip[:,:,np.newaxis] * (np.array(pheroA_colour)[np.newaxis,np.newaxis,:] - g) + amplifier * (
            ((np.where(pheroA_clip >= 0.05, 1-pheroA_clip, 0)[:,:,np.newaxis]) * np.array((96,96,96))[np.newaxis,np.newaxis,:]) +
            ((np.where((pheroA_clip > 0.0) & (pheroA_clip < 0.05), 1-pheroA_clip, 0)[:,:,np.newaxis]) * np.array((64,64,64))[np.newaxis,np.newaxis,:])
        )
        g += pheroB_clip[:,:,np.newaxis] * (np.array(pheroB_colour)[np.newaxis,np.newaxis,:] - g) + amplifier * (
            ((np.where(pheroB_clip >= 0.05, 1-pheroB_clip, 0)[:,:,np.newaxis]) * np.array((96,96,96))[np.newaxis,np.newaxis,:]) +
            ((np.where((pheroB_clip > 0.0) & (pheroB_clip < 0.05), 1-pheroB_clip, 0)[:,:,np.newaxis]) * np.array((64,64,64))[np.newaxis,np.newaxis,:])
        )
        
        #_ Add environment items to canvas
        g[states == Cell.item.FOOD] = np.array((204,204,0))[np.newaxis,np.newaxis,:]
        g[states == Cell.item.NEST] = np.array((153, 76,0))[np.newaxis,np.newaxis,:]
        g[states == Cell.item.WALL] = np.array((102, 51,0))[np.newaxis,np.newaxis,:]

        ###_ Render Agents _###

        for agent in self.env.colony.agents:
            # if agent.tracked:
            #     g[tuple(agent.get_pos())] = (255,0,0)
            # else:
                g[tuple(agent.get_pos())] = np.abs(self.bg - (255,255,255))#np.abs(self.bg - ((153,255,153) if agent.state == 1 else (153,153,255)))
                    

        ###_ Blit canvas and add info HUD _###
        surf = pg.surfarray.make_surface(g)
        surf = pg.transform.scale_by(surf,(
            self.screen.get_width() / self.env.grid.shape[0],
            self.screen.get_height() / self.env.grid.shape[1]
        ))

        self.screen.blit(surf, (0,0))
        # self.screen.blit(fps_txt,fps_txt_rect)
        # self.screen.blit(ups_txt,ups_txt_rect)
        # self.screen.blit(rt_txt,rt_txt_rect)
        self.screen.blit(it_txt,it_txt_rect)
        # self.screen.blit(dt_txt,dt_txt_rect)
        
        pg.display.update()

    def main(self):
        """
        Main loop for the visualiser. This will run until the user closes the window.
        """
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