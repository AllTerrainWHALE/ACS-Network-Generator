import numpy as np
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

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
            layers:list[int]=[10,8,4,1],
            genotype:torch.FloatTensor=None,

            activation_func:str="relu",

            wb_magnitude:float=.1
    ) -> None:
        super(AgentNN, self).__init__()

        self.layers = layers
        self.batch_norms = [nn.BatchNorm1d(c, dtype=torch.double).to(AgentNN.device) for c in layers[1:-1]]

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

        if activation_func == "relu": self.activation_func = nn.LeakyReLU(0.01)
        elif activation_func == "sigmoid": self.activation_func = nn.Sigmoid()
        elif activation_func == "tanh": self.activation_func = nn.Tanh()

        # print(self.weights, self.biases, sep='\n\n')

    def predict(self, x):
        # for w,b in zip(self.weights,self.biases):
        #     # print((x, x.shape), (w, w.shape), (b, b.shape), sep='\n', end='\n\n')
        #     x = self.activation_func(x @ w + b)
        # return x * pi
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i != len(self.weights) - 1:  # Apply activation to all but last layer
                # print(x.shape)
                x = self.activation_func(x)#self.batch_norms[i](x))
        return x  # Remove scaling by pi
    
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
        
        # Prepare loss function
        loss_fn = nn.HuberLoss(reduction='none')
        losses = np.zeros((random_states,epochs), dtype=np.float16)

        # Prepare optimizer and scheduler
        optimizer = optim.Adam(list(agent.weights)+list(agent.biases), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2, eta_min=1e-6)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000)

        # print(optimizer.param_groups)

        # Prepare random training conditions
        rand_general = torch.cat([
            torch.cat((
                (lambda b: torch.FloatTensor([np.sin(b),np.cos(b)]))(torch.rand(1) * 2*pi),       # Bearing
                torch.IntTensor([i%2]),     # State
                torch.rand(8, dtype=torch.float64) * 0xFFFFFFFFFFFFFFFF  # Neighbouring cells
            )).unsqueeze(0) for i in range(int((random_states-1) * 0.7))
        ])   # Shape: [random_states, 10]

        rand_equal_surr = torch.cat([
            torch.cat((
                (lambda b: torch.FloatTensor([np.sin(b),np.cos(b)]))(torch.rand(1) * 2*pi),       # Bearing
                torch.IntTensor([i%2]),     # State
                torch.ones(8, dtype=torch.float64) * torch.rand(1, dtype=torch.float64) * 0xFFFFFFFFFFFFFFFF # Neighbouring cells
            )).unsqueeze(0) for i in range(int((random_states-1) * 0.3))
        ])

        empty_surr = torch.cat([
            torch.cat([
                (lambda b: torch.FloatTensor([np.sin(b),np.cos(b)]))(torch.rand(1) * 2*pi), # Bearing
                torch.IntTensor([i%2]),    # State
                torch.zeros(8, dtype=torch.float64)              # Neighbouring cells
            ]).unsqueeze(0) for i in range(2)
        ])

        inputs = torch.cat([rand_general, rand_equal_surr, empty_surr]).to(AgentNN.device)
        inputs = inputs[torch.randperm(inputs.size(0))]

        # Prepare agent attributes to match the conitions
        bearings = inputs[:, 0:2].cpu().numpy()
        agent.bearing = np.arctan2(bearings[:, 0], bearings[:, 1])
        agent.state = inputs[:, 2].cpu().numpy()

        # Surrounding input for `follow_phero()`
        surrounding = torch.cat([
            inputs[:, 3:7],
            torch.zeros((random_states, 1), device=AgentNN.device),  # insert dummy cells
            inputs[:, 7:]
        ], dim=1).cpu().numpy() # Shape: [random_states, 9]
        
        # Calculate ground truths (y)
        delta_bearings = torch.DoubleTensor(agent.follow_phero(surrounding)[1])
        delta_bearings = (delta_bearings + pi) % (2*pi) - pi

        delta_bearings = 0.9 * delta_bearings + 0.1 * np.roll(delta_bearings, shift=1)
        delta_bearings = np.clip(delta_bearings, -np.pi/4, np.pi/4)

        y = delta_bearings.to(AgentNN.device) / pi

        # y = torch.DoubleTensor([[max(b, 0)/pi, max(-b, 0)/pi] for b in delta_bearings]).to(AgentNN.device)

        # Prepare neighour batch normalizing
        batch_norm_neighbours = torch.vmap(lambda x: torch.vmap(Cell.normalize)(x))

        if progress_bar:
            print(
                f"> Training base genotype...",
                f"\t{'Activation Func':<15} : {agent.activation_func.__class__.__name__}",
                f"\t{'Epochs':<15} : {epochs:,}",
                f"\t{'Learning Rate':<15} : {lr}",
                f"\t{'States':<15} : {random_states:,}",
                f"\t{'Device':<15} : {AgentNN.device}",

                sep='\n',
                end='\n\n'
            )
            printProgressBar(
                iteration=0, total=epochs-1,
                prefix=f'\tEpoch: {0:0{len(str(epochs))}}', suffix=f'ETA: 0:00:00',
                length=50, printEnd='\n'
            )
            print(f'\t\tLoss: 0.00e+0, Max: 0.00e+0', end='\x1b[1A')
        
        dts = [] # list of epoch durations
        batch_size = 256 #inputs.size(0)
        for e in range(epochs):
            timer = datetime.now()

            # Shuffle data and split into batches
            indices = torch.randperm(inputs.size(0))
            for i in range(0, inputs.size(0), batch_size):
                batch_indices = indices[i:i+batch_size]
                x_batch = inputs[batch_indices]
                y_batch = y[batch_indices]

                # x_batch = inputs.clone()

                # Normalize neighbour cells
                x_batch[:, 3:] = batch_norm_neighbours(x_batch[:, 3:])

                # Clear optimizer gradients
                optimizer.zero_grad()

                # Get predicted (y_hat)
                y_hat_batch = agent.predict(x_batch).double().squeeze(1)

                # Calculate loss
                each_loss = loss_fn(y_hat_batch, y_batch)
                mean_loss = each_loss.mean()
                # max_loss = torch.max(torch.mean(each_loss, dim=-1))

                losses[batch_indices,e] = each_loss.detach().cpu().numpy()#np.mean(each_loss.detach().cpu().numpy(), axis=-1, dtype=np.float16)

                # Calculate gradients
                mean_loss.backward()#gradient=torch.ones_like(each_loss))
                nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)

                # Update weights & biases
                optimizer.step()
                scheduler.step()

            dts.append((datetime.now() - timer).total_seconds())
            dts = dts[-1000:]

            if progress_bar and e % max(1,round(100/random_states)) == 0:
                printProgressBar(
                    iteration=e, total=epochs-1,
                    prefix=f'\tEpoch: {e+1:0{len(str(epochs))}}',
                    suffix=f'ETA: {str(timedelta(seconds=(epochs - e) * (sum(dts)/len(dts)))).split(".")[0]}   ',
                    length=50, printEnd='\n'
                )
                print('\t\t'+
                     'Loss: {',
                        f'Mean: {Decimal(np.sum(losses[:,e], dtype=float)/len(losses[:,e])):.4e},',
                        f'Max: {Decimal(float(np.max(losses[:,e]))):.4e}',
                     '},',
                    f'lr: {Decimal(scheduler.get_last_lr()[0]):.4e}',
                    
                    end='\x1b[1A'
                )

        agent.genotype = AgentNN.get_genotype_from_agent(agent)

        if progress_bar:
            # printProgressBar(
            #     iteration=e, total=epochs-1,
            #     prefix=f'\tEpoch: {epochs}', suffix=f'Loss: {Decimal(sum(losses[:,e])/len(losses[:,e])):.2e}, Max: {Decimal(np.max(losses[:,e])):.2e}',
            #     length=50, printEnd='\n'
            # )
            print('\n')
        # print(*list(zip(y_hat.detach().cpu().tolist(), y.detach().cpu().tolist())), sep='\n')
        print("\tTraining complete!")
        
        if loss_graph:
            x_vals = np.arange(losses.shape[1])[::100]
            for y_vals in losses[:, ::100]:
                plt.plot(x_vals, y_vals)

            y_mean_vals = np.mean(losses[:, ::100], axis=0)
            plt.plot(x_vals,y_mean_vals, linestyle='dashed', color='r')

            plt.xlabel('Epochs')
            plt.ylabel('Losses')

            plt.title((
                f"af(x): {agent.activation_func.__class__.__name__} | "+
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
            folder_name = f"{curr_time}_{Decimal(np.sum(losses[:,e], dtype=float)/len(losses[:,e])):.2e}"

            print("\t\t", end="")
            if make_dir(save_dir+folder_name):
            
                # Save loss graph (if loss graph is shown)
                if loss_graph:
                    plt.savefig(f"{save_dir}/{folder_name}/Figure_1.png")
                    print(f"\t\tFile '{save_dir}/{folder_name}/Figure_1.png' saved successfully.")

                # Save genotype and parameters
                d = {
                    "afunc"     : agent.activation_func.__class__.__name__,
                    "epochs"    : epochs,
                    "lr"        : lr,
                    "states"    : random_states,

                    "mean_loss" : np.sum(losses[:,e], dtype=float)/len(losses[:,e]),
                    "max_loss"  : float(np.max(losses[:,e])),

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