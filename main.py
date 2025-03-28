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

def environment_updater(env, iters, stop_event=None):
    #// durations = [0]
    iter = 0
    # x,y = 250,125
    while iter != iters:
        if stop_event.is_set(): break

        #// start = time()
        env.update()
        #// durations.append(time() - start)

        iter += 1
    
    if stop_event:
        stop_event.set()

    #// print(*durations, sep='\n', end='\n\n')
    #// average_dur = sum(durations[2:]) / len(durations[2:])
    #// ups = 1 / average_dur
    #//
    #// print(
    #//
    #//     f" " + f"_"*37 + f" ",
    #//     f"|{'RESULTS':^37}|",
    #//     f"|" + f" "*37 + f"|",
    #//
    #//     f"|{'Average':>17} : {round(average_dur, 6):<17}|",
    #//     f"|{'UPS':>17} : {round(ups, 0):<17}|",
    #//     f"|" + f"_"*37 + f"|",
    #//
    #//     sep='\n',
    #//     end='\n\n'
    #// )

def environment_visualiser(env, res, stop_event):
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
    parser.add_argument('--timestep', '-t', type=float, default=0.01, help="Time step for each update")

    parser.add_argument('--prefab', '-p', type=str, default="default", help="Prefab to load")

    parser.add_argument('--diffusion_rate', '-df', type=float, default=DIFFUSION_RATE, help="Pheromone diffusion rate")
    parser.add_argument('--evaporation_rate', '-ef', type=float, default=EVAPORATION_RATE, help="Pheromone evaporation rate")

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

    ###_ Initialize Environment and Colony _###
    env = Environment(args.env_res,
        food_source_positions=food_poss,
        diffusion_rate=args.diffusion_rate, evaporation_rate=args.evaporation_rate,
        timestep=args.timestep
    )
    
    env.colony = Colony(
        env, colony_pos, args.agents
    )


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