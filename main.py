import numpy as np
import threading as th
import argparse

from time import time, sleep
from math import pi

import json

from src.environment import Environment, Visualiser, DIFFUSION_RATE, EVAPORATION_RATE
from src.colony import Colony
from src.agent import Agent
from src.cell import Cell
from src.utils import printProgressBar, bcolors as bc

from colorama import init
init()

#_ Function to update and maintain the environment on an isolated thread
def environment_updater(env, iters, stop_event=None):
    """
    Function to update the environment and colony.

    The environment thread is halted when the visualiser is closed and vise versa (assuming a visualiser exists).
    """
    #// durations = [0]
    iter = 0
    # x,y = 250,125
    while iter != iters:
        if stop_event.is_set(): break
        env.update()

        iter += 1
    
    if stop_event:
        stop_event.set()

#_ Function to generate and maintain environment visualiser on an isolated thread
def environment_visualiser(env, res, stop_event):
    """
    Function to generate and maintain the environment visualiser.

    The environment thread is halted when the visualiser is closed and vise versa.
    """
    env_vis = Visualiser(env, fps=15, screen_res=res)

    try:
        env_vis.main()
    except Exception as e:
        print(e)
    finally:
        stop_event.set()


if __name__ == '__main__':

    #_ Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_res', '-er', type=list[int,int], default=[200,200], help="Environment resolution (rows, columns)")
    parser.add_argument('--env_vis_res', '-evr', type=list[int,int], default=[1000,1000], help="Visual environment resolution (rows, columns)")

    parser.add_argument('--agents', '-a', type=int, default=-1, help="Number of agents. Default is 25 per node")
    parser.add_argument('--updates', '-u', type=int, default=-1, help="Number of updates. Default is infinite")
    parser.add_argument('--timestep', '-t', type=float, default=0.01, help="Timestep for each update in seconds")

    parser.add_argument('--prefab', '-p', type=str, default="default", help="The chosen prefab to load into the environment. The prefab contains nest and food source positions")

    parser.add_argument('--diffusion_rate', '-df', type=float, default=DIFFUSION_RATE, help="Pheromone diffusion rate (0-1)")
    parser.add_argument('--evaporation_rate', '-ef', type=float, default=EVAPORATION_RATE, help="Pheromone evaporation rate (0-1)")

    args = parser.parse_args()

    #_ Load prefab
    prefabs = None
    try:
        with open('prefabs.json', 'r') as file:
            prefabs = dict(json.load(file))
            print("> prefabs.json successfully loaded")
    except FileNotFoundError:
        print("> prefabs.json not found. Loading with default prefab")
    except json.JSONDecodeError:
        print("> Error decoding prefabs file. Loading with default prefab")
    finally:
        if prefabs == None:
            prefabs = {}

    if args.prefab not in prefabs.keys():
        print(f"> Prefab '{args.prefab}' not found. Loading with default prefab")
        args.prefab = "default"

    colony_pos = prefabs[args.prefab]['nest']
    food_poss = prefabs[args.prefab]['food']

    #_ Configure agent count
    if args.agents < 0: # Default agent count
        # 25 agents for every node (nest + food source)
        args.agents = (len(food_poss) + 1) * 25

    #_ Print settings in a pretty way
    print("",
        
        f" " + f"_"*37 + f" ",
        f"|{'SETTINGS':^37}|",
        f"|" + f" "*37 + f"|",
        
        f"|{'Environment Size':>17} : {' x '.join(map(str, args.env_res)):<17}|",
        f"|{'Vis. Resolution':>17} : {' x '.join(map(str, args.env_vis_res)):<17}|",
        f"|{'Prefab':>17} : {args.prefab:<17}|",
        f"|" + f" "*37 + f"|",
        f"|{'Agents':>17} : {args.agents:<17}|",
        f"|{'Updates':>17} : {args.updates:<17}|",
        f"|{'Time Step':>17} : {args.timestep:<17}|",
        f"|" + f" "*37 + f"|",
        f"|{'Diffusion Rate':>17} : {args.diffusion_rate:<17}|",
        f"|{'Evaporation Rate':>17} : {args.evaporation_rate:<17}|",
        f"|" + f"_"*37 + f"|",

        sep='\n',
        end='\n\n'
    )
    #// print(f'{bc.FAIL}EXPERIMENT WITH MOVING FOOD SOURCES{bc.ENDC}')

    ###_ Initialize Environment and Colony _###
    env = Environment(args.env_res,
        food_source_positions=food_poss,
        diffusion_rate=args.diffusion_rate, evaporation_rate=args.evaporation_rate,
        timestep=args.timestep
    )
    
    env.colony = Colony(
        env, colony_pos, args.agents
    )

    #_ User input for visualisation
    answer = None
    while answer not in ['y','n','']:
        answer = input("> Do you want the visual environment? ([y]/n) ").lower().strip(' ')
        print()

        if answer not in ['y','n','']:
            print("> Invalid response\n")

    if answer in ['y','']:
        stop_event = th.Event()

        # Create threads
        updater_thread = th.Thread(target=environment_updater, args=(env,args.updates,stop_event,))
        visualiser_thread = th.Thread(target=environment_visualiser, args=(env,args.env_vis_res,stop_event,))

        # Start threads
        updater_thread.start()
        visualiser_thread.start()

        # Join threads (await completion of both threads)
        updater_thread.join()
        visualiser_thread.join()

    else:
        if args.updates < 0:
            print(f"> A count of {args.updates} updates will result in an infinite loop. Please set an alternative count")
        else:
            environment_updater(env, args.updates)