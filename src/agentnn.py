import numpy as np
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from math import pi
from typing import TYPE_CHECKING
from decimal import Decimal
from datetime import datetime, timedelta
from time import time

from src.cell import Cell
from src.utils import printProgressBar, make_dir

if TYPE_CHECKING:
    from src.agent import Agent  # Only imported for type checking

class AgentNN (nn.Module):

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
            self,
            layers:list[int]=[10,8,4,2],
            genotype:torch.FloatTensor=None,

            activation_func:str="relu",

            wb_magnitude:float=.01
    ) -> None:
        super(AgentNN, self).__init__()

        self.layers = layers

        # Prepare genotype, weights and biases
        if genotype == None:
            self.weights = nn.ParameterList(
                [nn.Parameter(torch.rand(layers[i], layers[i + 1], dtype=torch.float64)*wb_magnitude) for i in range(len(layers) - 1)]
            )
            self.biases = nn.ParameterList(
                [nn.Parameter(torch.rand(1, layers[i + 1], dtype=torch.float64)*wb_magnitude) for i in range(len(layers) - 1)]
            )

            self.genotype = AgentNN.get_genotype_from_agent(self)

        elif genotype != None:
            self.genotype = torch.FloatTensor(genotype)

            self.weights, self.biases = AgentNN.get_wb_from_agent(self)

        else:
            raise Exception("What da fuq is `genotype`?")
        
        # Prepare activation function
        activation_func = activation_func.lower()

        if activation_func == "relu": self.activation_func = torch.relu
        elif activation_func == "sigmoid": self.activation_func = torch.sigmoid
        elif activation_func == "tanh": self.activation_func = torch.tanh

        print(self.weights, self.biases, sep='\n\n')

    def predict(self, x):
        for w,b in zip(self.weights,self.biases):
            # print((x, x.shape), (w, w.shape), (b, b.shape), sep='\n', end='\n\n')
            x = self.activation_func(x @ w + b)
        return x * pi
    
    @staticmethod
    def train_follow_phero(
        agent:"Agent", epochs:int,
        
        random_states:int=100,
        lr:float=.001,
        
        progress_bar=True,
        loss_graph=False, save_loss_data=False, save_dir="training_figures/phero_following/"
    ) -> list[float]:
    
        # Move the agent's parameters to the GPU
        agent.to(AgentNN.device)
        
        loss_fn = nn.MSELoss(reduction='none')
        losses = np.zeros((random_states,epochs))

        optimizer = optim.SGD(list(agent.weights)+list(agent.biases), lr=lr, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5000, eps=1e-8)

        # Prepare random training conditions
        rand_general = torch.cat([
            torch.cat((
                torch.rand(1) * 2*pi,       # Bearing
                torch.IntTensor([i%2]),     # State
                torch.rand(8, dtype=torch.float64) * 0xFFFFFFFFFFFFFFFF  # Neighbouring cells
            )).unsqueeze(0) for i in range(int(random_states * 0.7))
        ])   # Shape: [random_states, 10]

        rand_empty_surr = torch.cat([
            torch.cat((
                torch.rand(1) * 2*pi,       # Bearing
                torch.IntTensor([i%2]),     # State
                torch.ones(8, dtype=torch.float64) * torch.rand(1, dtype=torch.float64) * 0xFFFFFFFFFFFFFFFF # Neighbouring cells
            )).unsqueeze(0) for i in range(int(random_states * 0.3))
        ])

        inputs = torch.cat([rand_general, rand_empty_surr]).to(AgentNN.device)

        if progress_bar:
            print(
                f"> Training base genotype...",
                f"\t{'Activation Func':<15} : {agent.activation_func.__name__}",
                f"\t{'Epochs':<15} : {epochs}",
                f"\t{'Learning Rate':<15} : {lr}",
                f"\t{'States':<15} : {random_states}",

                sep='\n',
                end='\n\n'
            )
            printProgressBar(
                iteration=0, total=epochs-1,
                prefix=f'\tEpoch: {0:0{len(str(epochs))}}', suffix=f'Loss: 0.00e+0, Max: 0.00e+0',
                length=50
            )

        dts = []
        old_params = (
            list(map(lambda p: p.data, agent.weights.parameters())),
            list(map(lambda p: p.data, agent.biases.parameters()))
            )
        
        for e in range(epochs):
            timer = datetime.now()
            # # Create random conditions
            # x = torch.cat((
            #     torch.rand(1) * 2*pi,                 # Bearing
            #     torch.randint(0,2,(1,)),              # State
            #     torch.rand(8) * 0xFFFFFFFFFFFFFFFF    # Neighbouring cells
            # ))
            x_batch = inputs.clone()

            # Surrounding input for `follow_phero()`
            surrounding = torch.cat([
                x_batch[:, 2:6],
                torch.zeros((random_states, 1), device=AgentNN.device),  # insert dummy cells
                x_batch[:, 6:]
            ], dim=1).cpu().numpy() # Shape: [random_states, 9]

            # Prepare agent attributes to match the conitions
            agent.bearing = x_batch[:, 0].cpu().numpy()
            agent.state = x_batch[:, 1].cpu().numpy()

            # Normalize neighbour cells
            x_batch[:, 2:] = torch.DoubleTensor([
                list(map(Cell.normalize, row.cpu().numpy())) for row in x_batch[:, 2:]
            ]).to(AgentNN.device)

            # x_batch = x_batch.unsqueeze(1)

            # Clear optimizer gradients
            optimizer.zero_grad()

            # Get predicted (y_hat) and expected (y)
            y_hat = agent.predict(x_batch).double()

            delta_bearings = agent.follow_phero(surrounding)[1]#np.array([agent.follow_phero(s)[1] for s in surrounding])
            delta_bearings = (delta_bearings + pi) % (2*pi) - pi
            y = torch.DoubleTensor([[max(b, 0), max(-b, 0)] for b in delta_bearings]).to(AgentNN.device)

            # Calculate loss
            each_loss = loss_fn(y_hat, y)
            # max_loss = torch.max(torch.mean(each_loss, dim=-1))

            losses[:,e] = np.mean(each_loss.detach().cpu().numpy(), axis=-1)

            # Calculate gradients
            each_loss.mean().backward()#gradient=torch.ones_like(each_loss))

            # Update weights & biases
            optimizer.step()
            # scheduler.step(np.mean(losses[:,e]))

            dts.append((datetime.now() - timer).total_seconds())
            dts = dts[-1000:]

            if progress_bar and e % max(1,round(100/random_states)) == 0:
                printProgressBar(
                    iteration=e, total=epochs-1,
                    prefix=f'\tEpoch: {e+1:0{len(str(epochs))}}, ETA: {str(timedelta(seconds=(epochs - e) * (sum(dts)/len(dts)))).split(".")[0]}',
                    suffix=f'Loss: {Decimal(sum(losses[:,e])/len(losses[:,e])):.2e}, Max: {Decimal(np.max(losses[:,e])):.2e}',
                    length=50
                )

        agent.genotype = AgentNN.get_genotype_from_agent(agent)

        # if progress_bar: printProgressBar(
        #     iteration=e, total=epochs-1,
        #     prefix=f'\tEpoch: {epochs}', suffix=f'Loss: {Decimal(sum(losses[:,e])/len(losses[:,e])):.2e}, Max: {Decimal(np.max(losses[:,e])):.2e}',
        #     length=50, printEnd='\n'
        # )
        # print(*list(zip(y_hat.detach().cpu().tolist(), y.detach().cpu().tolist())), sep='\n')
        print("\tTraining complete!")
        
        if loss_graph:
            x_vals = np.arange(losses.shape[1])
            for y_vals in losses:
                plt.plot(x_vals, y_vals)

            plt.xlabel('Epochs')
            plt.ylabel('Losses')

            plt.title((
                f"af(x): {agent.activation_func.__name__} | "+
                f"Epochs: {epochs} | "+
                f"LR: {lr} | "+
                f"States: {random_states} "),
                fontsize=10
            )

            plt.show(block=False)

        if save_loss_data:
            print("\n\tSaving loss data...")

            # Create save directory
            curr_time = datetime.now().strftime("%Y-%m-%d_%H%M")
            folder_name = f"{curr_time}_{Decimal(sum(losses[:,e])/len(losses[:,e])):.2e}"

            print("\t\t", end="")
            if make_dir(save_dir+folder_name):
            
                # Save loss graph (if loss graph is shown)
                if loss_graph:
                    plt.savefig(f"{save_dir}/{folder_name}/Figure_1.png")
                    print(f"\t\tFile '{save_dir}/{folder_name}/Figure_1.png' saved successfully.")

                # Save genotype and parameters
                d = {
                    "afunc"     : agent.activation_func.__name__,
                    "epochs"    : epochs,
                    "lr"        : lr,
                    "states"    : random_states,

                    "mean_loss" : sum(losses[:,e])/len(losses[:,e]),
                    "max_loss"  : np.max(losses[:,e]),

                    "genotype"  : agent.genotype.detach().tolist()
                }

                with open(f"{save_dir}/{folder_name}/settings.json", "w") as outfile:
                    json.dump(d, outfile, indent=4)
                    print(f"\t\tFile '{save_dir}/{folder_name}/settings.json' saved successfully.")
                
                print("\tSave successful!")

            else:
                print("\tSave failed! Directory could not be found or made")
        return agent.genotype, losses


    @staticmethod
    def get_genotype_from_wb(Ws:nn.ParameterList, bs:nn.ParameterList) -> torch.FloatTensor:
        genotype = []
        for Wss,bss in zip(Ws,bs):
            genotype += [
                *[w.item() for Ws in Wss for w in Ws],
                *[b.item() for bs in bss for b in bs]
            ]
        return torch.FloatTensor(genotype)

    @classmethod
    def get_genotype_from_agent(cls, agent:"AgentNN") -> torch.FloatTensor:
        return cls.get_genotype_from_wb(agent.weights, agent.biases)

    @staticmethod
    def get_wb_from_genotype(genotype:torch.Tensor, layers:list[int]) -> tuple[nn.ParameterList,nn.ParameterList]:

        Wss = []
        bss = []
        start_idx = 0
        for i in range(1,len(layers)):
            
            middle_idx = start_idx + layers[i-1]*layers[i]
            end_idx = middle_idx + layers[i]

            Ws = genotype[start_idx:middle_idx]
            bs = genotype[end_idx-layers[i]:end_idx]

            #// print(f"Ws: {Ws}\nbs: {bs}", end='\n\n')

            Ws = Ws.reshape((layers[i-1], layers[i]))
            bs = bs.reshape(1,layers[i])

            #// print(f"Ws: {Ws}\nbs: {bs}", end='\n\n')

            Wss.append(nn.Parameter(Ws.double()))
            bss.append(nn.Parameter(bs.double()))

            start_idx = end_idx

        #// print(Wss, bss, sep='\n\n', end='\n\n')

        return nn.ParameterList(Wss), nn.ParameterList(bss)
    
    @classmethod
    def get_wb_from_agent(cls, agent:"AgentNN") -> tuple[nn.ParameterList,nn.ParameterList]:
        return cls.get_wb_from_genotype(agent.genotype, agent.layers)