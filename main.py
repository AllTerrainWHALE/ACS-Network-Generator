import numpy as np
import threading as th

from src.environment import Environment, Visualiser
from src.cell import Cell
from time import time, sleep

def environment_updater(env, iters, stop_event=None):
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

        f" " + f"_"*35 + f" ",
        f"|{'RESULTS':^35}|",
        f"|" + f" "*35 + f"|",

        f"|{'Average':>16} : {round(average_dur, 6):<16}|",
        f"|{'UPS':>16} : {round(ups, 0):<16}|",
        f"|" + f"_"*35 + f"|",

        sep='\n',
        end='\n\n'
    )

def environment_visualiser(env, stop_event):

    env_vis = Visualiser(env, fps=30, screen_res=(1440,1440))
    env_vis.main()

    stop_event.set()


if __name__ == '__main__':
    grid_res = (500,500)
    agents = 1
    updates = -1

    print("",
        
        f" " + f"_"*35 + f" ",
        f"|{'SETTINGS':^35}|",
        f"|" + f" "*35 + f"|",
        
        f"|{'Resolution':>16} : {' x '.join(map(str, grid_res)):<16}|",
        f"|{'Agents':>16} : {agents:<16}|",
        f"|{'Updates':>16} : {updates:<16}|",
        f"|" + f"_"*35 + f"|",

        sep='\n',
        end='\n\n'
    )

    env = Environment(grid_res, agents)
    # env.grid[100:150,100:150] = Cell.setState(env.grid[100:150,100:150], 1)
    # env.grid[2,2] = Cell.setPheroB(env.grid[2,2], 1000)

    print(f"{'INITIAL GRID':^34}")
    print(np.array([[Cell.getPheroA(cx) for cx in cy] for cy in env.grid]), end='\n\n')
    # print(np.array([[Cell.getPheroB(cx) for cx in cy] for cy in env.grid]), end='\n\n')

    answer = None
    while answer not in ['y','n']:
        answer = input("Do you want the visual environment? (Y/N)\n").lower()
        print()

        if answer not in ['y','n']:
            print("Invalid response\n")

    if answer == 'y':
        stop_event = th.Event()

        # Create threads
        updater_thread = th.Thread(target=environment_updater, args=(env,updates,stop_event,))
        visualiser_thread = th.Thread(target=environment_visualiser, args=(env,stop_event,))

        # Start threads
        updater_thread.start()
        visualiser_thread.start()

        # Join threads (await completion of both threads)
        updater_thread.join()
        visualiser_thread.join()

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