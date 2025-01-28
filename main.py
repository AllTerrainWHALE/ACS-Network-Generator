import numpy as np
import threading as th

import torch
import torch.nn as nn
import torch.optim as optim

from time import time, sleep
from math import pi

from src.environment import Environment, Visualiser
from src.colony import Colony
from src.agent import Agent
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
    agents_per_colony = 20
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
    epochs = 10000
    examples = 100
    test_agent = Agent()
    train_agent = Agent()

    print("Training Pheromone Following Genotype...")
    printProgressBar(
            iteration=0, total=epochs-1,
            prefix='Progress', suffix=f'Loss: 0',
            length=50
        )
    
    # Prepare random training conditions
    inputs = [torch.cat((
        torch.rand(1) * 2*pi,       # Bearing
        torch.randint(0,2,(1,)),    # State
        torch.rand(8) * 0x7FFFFFFF  # Neighbouring cells
    )) for i in range(examples)]

    # Ws, bs = [], []
    # for i in range(1, len(test_agent.layers)):
    #     Ws.append(nn.Parameter(torch.rand(test_agent.layers[i-1], test_agent.layers[i])))
    #     bs.append(nn.Parameter(torch.rand(1, test_agent.layers[i])))

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD((*test_agent.Ws, *test_agent.bs), lr=.001)

    for e in range(epochs):
        # Create random conditions
        # x = torch.cat((
        #     torch.rand(1) * 2*pi,       # Bearing
        #     torch.randint(0,2,(1,)),    # State
        #     torch.rand(8) * 0x7FFFFFFF  # Neighbouring cells
        # ))

        for i in inputs:

            surrounding = np.array((
                *i[2:6],
                0,  # insert dummy cell
                *i[6:]
            ))

            # Prepare testing and training agents with random condition
            test_agent.bearing = float(i[0])
            train_agent.bearing = float(i[0])
            
            test_agent.state = int(i[1])
            train_agent.state = int(i[1])

            
            x = i.unsqueeze(0).float()

            # Clear optimizer gradients
            optimizer.zero_grad()

            # Get testing and training results
            y_hat = test_agent.predict(x)

            translation, bearing = train_agent.follow_phero(surrounding)
            bearing = (bearing + pi) % (2*pi) - pi
            y = torch.tensor([[max(bearing,0), max(-bearing,0)]])

            # Calculate loss
            loss = loss_fn(y_hat, y)

            # Calculate gradients
            grad = loss.backward()

            # Update weights & biases
            optimizer.step()

            # print(optimizer.param_groups[0]['params'])

            # Update test Agent's genotype
            Ws_and_bs = optimizer.param_groups[0]['params']

            new_genotype = []
            for n in range(int(len(Ws_and_bs)/2)):
                Wss = Ws_and_bs[n]
                bss = Ws_and_bs[int(len(Ws_and_bs)/2) + n]

                # Ws = [w.item() for Ws in Wss for w in Ws]
                # bs = [b.item() for bs in bss for b in bs]

                #// print([w.item() for W in Ws for w in W], end='\n-----------\n')
                new_genotype += [
                    *[w.item() for Ws in Wss for w in Ws],
                    *[b.item() for bs in bss for b in bs]
                ]
            test_agent.genotype = np.array(new_genotype)

        if e % 1 == 0:
            printProgressBar(
                iteration=e, total=epochs-1,
                prefix='Progress', suffix=f'Loss: {loss}',
                length=50
            )
    

    exit()


    env = Environment(env_res)
    env.colonies = [Colony(env, (int(env_res[0]/2),int(env_res[1]/2)), agents_per_colony)]

    # env.grid[100:150,100:150] = Cell.setState(env.grid[100:150,100:150], 1)
    # env.grid[2,2] = Cell.setPheroB(env.grid[2,2], 1000)

    # print(f"{'INITIAL GRID':^34}")
    # print(np.array([[Cell.getPheroA(cx) for cx in cy] for cy in env.grid]), end='\n\n')
    # print(np.array([[Cell.getPheroB(cx) for cx in cy] for cy in env.grid]), end='\n\n')

    answer = None
    while answer not in ['y','n']:
        answer = i("> Do you want the visual environment? (Y/N)\n").lower()
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
            environment_updater(env, updates)





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