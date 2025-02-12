import numpy as np
import threading as th
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from time import time, sleep
from math import pi

from src.environment import Environment, Visualiser
from src.colony import Colony
from src.agent import Agent
from src.agentnn import AgentNN
from src.cell import Cell
from src.utils import printProgressBar

from colorama import init
init()

def environment_updater(env, colony, iters, stop_event=None):
    durations = [0]
    iter = 0
    # x,y = 250,125
    while iter != iters:
        if stop_event and stop_event.is_set(): break

        start = time()
        env.update()
        # print(np.array([[Cell.getPheroA(cx) for cx in cy] for cy in env.grid[248:253,248:253]]), end='\n\n')
        # print(np.array([[Cell.getPheroB(cx) for cx in cy] for cy in env.grid]), end='\n\n')
        durations.append(time() - start)

        # y = min(env.grid.shape[1], max(0, y + 0.1))

        # env.grid[int(x),int(y)] = Cell.setPheroA(env.grid[int(x),int(y)], 0x7FFFFFFF)
        # env.grid[int(x),500-int(y)] = Cell.setPheroB(env.grid[int(x),500-int(y)], 0x7FFFFFFF)

        iter += 1
    
    if stop_event:
        stop_event.set()

    # print(*durations, sep='\n', end='\n\n')
    average_dur = sum(durations[2:]) / len(durations[2:])
    ups = 1 / average_dur

    print(

        f" " + f"_"*37 + f" ",
        f"|{'RESULTS':^37}|",
        f"|" + f" "*37 + f"|",

        f"|{'Average':>17} : {round(average_dur, 6):<17}|",
        f"|{'UPS':>17} : {round(ups, 0):<17}|",
        f"|" + f"_"*37 + f"|",

        sep='\n',
        end='\n\n'
    )

def environment_visualiser(env, res, stop_event):
    env_vis = Visualiser(env, fps=15, screen_res=res)

    try:
        env_vis.main()

    except Exception as e:
        print(e)

    finally:
        stop_event.set()


if __name__ == '__main__':

    env_res = (500,500)
    env_vis_res = (1000,1000)

    colonies = 1
    agents_per_colony = 1
    updates = -1

    print("",
        
        f" " + f"_"*37 + f" ",
        f"|{'SETTINGS':^37}|",
        f"|" + f" "*37 + f"|",
        
        f"|{'Environment Size':>17} : {' x '.join(map(str, env_res)):<17}|",
        f"|{'Vis. Resolution':>17} : {' x '.join(map(str, env_vis_res)):<17}|",
        f"|" + f" "*37 + f"|",
        f"|{'Colonies':>17} : {colonies:<17}|",
        f"|{'Agents / Colony':>17} : {agents_per_colony:<17}|",
        f"|{'Updates':>17} : {updates:<17}|",
        f"|" + f"_"*37 + f"|",

        sep='\n',
        end='\n\n'
    )


    ### Train Pheromone Following Genotype ###
    agent = Agent(activation_func='relu', genotype=None)
    AgentNN.train_follow_phero(agent, epochs=10000, random_states=10000, lr=0.001, loss_graph=True, save_loss_data=True)
    # input()

    # plt.ion()
    # fig, ax = plt.subplots(1,3, figsize=(15,5))

    # ax[0].set_title('relu')
    # ax[0].set_xlabel("Epochs")
    # ax[0].set_ylabel("Loss / State")
    # ax[0].set_ylim(0,5)

    # ax[1].set_title('sigmoid')
    # ax[1].set_xlabel("Epochs")
    # ax[1].set_ylabel("Loss / State")
    # ax[1].set_ylim(0,5)

    # ax[2].set_title('tanh')
    # ax[2].set_xlabel("Epochs")
    # ax[2].set_ylabel("Loss / State")
    # ax[2].set_ylim(0,5)

    # genotype = Agent().genotype

    # activation_funcs = ['tanh', 'relu','sigmoid']
    # epochs = 100000# epochs = [100, 1000, 10000, 100000]
    # lrs = [.1, .01, .001, .0001, .00001]
    # random_states = [100, 1000, 10000]
    
    # results = []

    # for rs in random_states:
    #     for lr in lrs:
    #         for func in activation_funcs:
    #             agent = Agent(genotype=genotype, activation_func=func)
    #             _, losses = AgentNN.train_follow_phero(agent, epochs=epochs, random_states=rs, lr=lr, loss_graph=False)

    #             results.append({
    #                 'func'   : func,
    #                 'lr' : lr,
    #                 'states' : rs,
    #                 'losses' : np.mean(losses, axis=0)
    #             })

    #             result = results[-1]

    #             x_vals = np.arange(epochs)
    #             y_vals = result['losses']

    #             if result['func'] == 'relu':
    #                 ax[0].plot(x_vals,y_vals,label=f"{result['lr']} | {result['states']}")
    #             elif result['func'] == 'sigmoid':
    #                 ax[1].plot(x_vals,y_vals,label=f"{result['lr']} | {result['states']}")
    #             elif result['func'] == 'tanh':
    #                 ax[2].plot(x_vals,y_vals,label=f"{result['lr']} | {result['states']}")

    #             ax[0].legend(title="LR | States")
    #             ax[1].legend(title="LR | States")
    #             ax[2].legend(title="LR | States")

    #             fig.canvas.draw()
    #             fig.canvas.flush_events()

    ### Initialize Environment and Colony ###
    env = Environment(env_res)
    env.colonies = [Colony(env, (int(env_res[0]/2),int(env_res[1]/2)), agents_per_colony, base_genotype=agent.genotype)]

    # env.grid[100:150,100:150] = Cell.setState(env.grid[100:150,100:150], 1)
    # env.grid[2,2] = Cell.setPheroB(env.grid[2,2], 1000)

    # print(f"{'INITIAL GRID':^34}")
    # print(np.array([[Cell.getPheroA(cx) for cx in cy] for cy in env.grid]), end='\n\n')
    # print(np.array([[Cell.getPheroB(cx) for cx in cy] for cy in env.grid]), end='\n\n')

    answer = None
    while answer not in ['y','n']:
        answer = input("> Do you want the visual environment? (Y/N)\n").lower()
        print()

        if answer not in ['y','n']:
            print("> Invalid response\n")

    if answer == 'y':
        stop_event = th.Event()

        # Create threads
        updater_thread = th.Thread(target=environment_updater, args=(env,colonies,updates,stop_event,))
        visualiser_thread = th.Thread(target=environment_visualiser, args=(env,env_vis_res,stop_event,))

        # Start threads
        updater_thread.start()
        visualiser_thread.start()

        # Join threads (await completion of both threads)
        updater_thread.join()
        visualiser_thread.join()

    else:
        if updates < 0:
            print(f"> A count of {updates} updates will result in an infinite loop. Please set an alternative count")
        else:
            environment_updater(env, colonies, updates)





    # print(f"{'INITIAL GRID':^34}")
    # print(np.array([[Cell.getPheroA(cx) for cx in cy] for cy in env.grid]), end='\n\n')
    # print(np.array([[Cell.getPheroB(cx) for cx in cy] for cy in env.grid]), end='\n\n')
    
    # durations = []
    # for _ in range(updates):
    #     start = time()
    #     env.update()
    #     # print(np.array([[Cell.getPheroA(cx) for cx in cy] for cy in env.grid]), end='\n\n')
    #     # print(np.array([[Cell.getPheroB(cx) for cx in cy] for cy in env.grid]), end='\n\n')
    #     durations.append(time() - start)

    # # print(*durations, sep='\n', end='\n\n')
    # average_dur = sum(durations[1:]) / len(durations[1:])
    # ups = 1 / average_dur

    # print(

    #     f" " + f"_"*35 + f" ",
    #     f"|{'RESULTS':^35}|",
    #     f"|" + f" "*35 + f"|",

    #     f"|{'Average':>16} : {round(average_dur, 6):<16}|",
    #     f"|{'UPS':>16} : {round(ups, 0):<16}|",
    #     f"|" + f"_"*35 + f"|",

    #     sep='\n',
    #     end='\n\n'
    # )