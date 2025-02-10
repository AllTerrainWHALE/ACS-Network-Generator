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
    agent = Agent(activation_func='sigmoid', genotype=[
        1.5994793176651,
        -0.5887181162834167,
        1.8452082872390747,
        -0.4271527826786041,
        -0.9843339323997498,
        -0.4893858730792999,
        -0.08916420489549637,
        1.7981210947036743,
        -0.34799063205718994,
        1.4517868757247925,
        1.9209879636764526,
        -2.689854145050049,
        -1.908308744430542,
        1.976743221282959,
        1.9344385862350464,
        3.783808469772339,
        -2.637248992919922,
        -0.08894317597150803,
        0.6040571331977844,
        1.2719080448150635,
        -2.677156925201416,
        -1.0113879442214966,
        -2.04963755607605,
        3.349307060241699,
        3.5627970695495605,
        2.6136531829833984,
        -1.228049874305725,
        5.672904968261719,
        2.5825133323669434,
        -2.154958963394165,
        1.950042486190796,
        -3.4500601291656494,
        0.5126467943191528,
        -0.47864988446235657,
        -5.265242099761963,
        -2.4949376583099365,
        0.28113868832588196,
        -2.8588485717773438,
        -1.371822714805603,
        -4.401430606842041,
        0.8701465129852295,
        2.552837371826172,
        -1.9897843599319458,
        0.04913748800754547,
        1.5288618803024292,
        3.274101972579956,
        0.7940444350242615,
        -0.8369008302688599,
        -1.1729350090026855,
        0.020558038726449013,
        -0.7026928663253784,
        3.675837993621826,
        -4.195079326629639,
        8.247188568115234,
        0.9936508536338806,
        -4.462460041046143,
        -5.681850433349609,
        -1.043513298034668,
        -1.6360877752304077,
        4.517660617828369,
        2.055774450302124,
        1.1198629140853882,
        0.3293137550354004,
        -4.271310806274414,
        0.4605240523815155,
        0.060206908732652664,
        1.929195761680603,
        1.0790531635284424,
        2.5595738887786865,
        -1.996397852897644,
        1.851943850517273,
        -2.04347562789917,
        -7.480911731719971,
        -2.483877658843994,
        -0.910585343837738,
        -3.501786708831787,
        -2.3097574710845947,
        0.16188722848892212,
        -3.8515098094940186,
        5.748566627502441,
        3.365689277648926,
        0.8604991436004639,
        -1.4866341352462769,
        -1.8420352935791016,
        1.4245306253433228,
        -3.1252527236938477,
        2.6839919090270996,
        -3.9746110439300537,
        2.698608160018921,
        2.724125862121582,
        1.627716064453125,
        3.4570374488830566,
        0.3681281507015228,
        0.2233685404062271,
        -2.6066691875457764,
        -1.1539181470870972,
        -5.919149398803711,
        -5.806264877319336,
        -4.264042377471924,
        -4.404114723205566,
        0.16412150859832764,
        -0.057252053171396255,
        6.9410014152526855,
        -0.6678537130355835,
        2.6911001205444336,
        2.8461716175079346,
        0.4808761477470398,
        4.6778740882873535,
        2.911864757537842,
        2.8897347450256348,
        4.09299373626709,
        1.3251454830169678,
        1.6571440696716309,
        1.76296865940094,
        0.14117811620235443,
        1.743276834487915,
        5.409188270568848,
        5.391726493835449,
        5.488471984863281,
        5.377103328704834,
        -4.242507457733154,
        -4.27301025390625,
        -7.417057514190674,
        -4.861070156097412,
        2.1638505458831787,
        -8.921096801757812,
        2.9654483795166016,
        -8.746376037597656,
        8.233811378479004,
        -5.831917762756348,
        6.7931952476501465,
        -7.2565388679504395,
        -10.323956489562988,
        7.699785232543945
    ])
    # AgentNN.train_follow_phero(agent, epochs=100000, random_states=100, lr=0.01, loss_graph=True, save_loss_data=True)
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