import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from math import pi
from typing import TYPE_CHECKING

from src.cell import Cell
from src.utils import printProgressBar

if TYPE_CHECKING:
    from src.agent import Agent  # Only imported for type checking

class AgentNN (nn.Module):

    def __init__(
            self,
            layers:list[int]=[10,4,2],
            genotype:torch.FloatTensor=None,

            wb_magnitude:float=.01
    ) -> None:
        super(AgentNN, self).__init__()

        self.layers = layers

        if not genotype:
            self.weights = nn.ParameterList(
                [nn.Parameter(torch.rand(layers[i], layers[i + 1])*wb_magnitude) for i in range(len(layers) - 1)]
            )
            self.biases = nn.ParameterList(
                [nn.Parameter(torch.rand(1, layers[i + 1])*wb_magnitude) for i in range(len(layers) - 1)]
            )

            self.genotype = AgentNN.get_genotype_from_agent(self)

        elif genotype:
            self.genotype = genotype

            self.weights, self.biases = AgentNN.get_wb_from_agent(self)

    def predict(self, x):
        for w,b in zip(self.weights,self.biases):
            x = torch.tanh(x @ w + b)
        return x
    


    @staticmethod
    def train_follow_phero(
        agent:"Agent", epochs:int,
        
        random_states:int=100,
        lr:float=.001,
        
        progress_bar=False, loss_graph=False
    ) -> list[float]:
        
        loss_fn = nn.MSELoss()
        losses = np.zeros((random_states,epochs))

        optimizer = optim.SGD((*agent.weights.parameters(),)+(*agent.biases.parameters(),), lr=lr)

        # Prepare random training conditions
        inputs = [torch.cat((
            torch.rand(1) * 2*pi,               # Bearing
            torch.randint(0,2,(1,)),            # State
            torch.rand(8) * 0xFFFFFFFFFFFFFFFF  # Neighbouring cells
        )) for i in range(random_states)]

        if progress_bar: printProgressBar(
            iteration=0, total=epochs-1,
            prefix='Progress', suffix=f'Loss: 0',
            length=50
        )
        
        for e in range(epochs):
            # # Create random conditions
            # x = torch.cat((
            #     torch.rand(1) * 2*pi,                 # Bearing
            #     torch.randint(0,2,(1,)),              # State
            #     torch.rand(8) * 0xFFFFFFFFFFFFFFFF    # Neighbouring cells
            # ))
            for i in range(len(inputs)):
                x = inputs[i].detach().clone()

                # Surrounding input for `follow_phero()`
                surrounding = np.array((
                    *x[2:6],
                    0,  # insert dummy cell
                    *x[6:]
                ))

                # Prepare agent attributes to match the conitions
                agent.bearing = float(x[0])
                agent.state = int(x[1])

                # Normalize neighbour cells
                x[2:] = torch.FloatTensor(list(map(Cell.normalize, x[2:])))

                x = x.unsqueeze(0)#.float()

                # Clear optimizer gradients
                optimizer.zero_grad()

                # Get testing and training results
                y_hat = agent.predict(x).float()

                bearing = agent.follow_phero(surrounding)[1]
                bearing = (bearing + pi) % (2*pi) - pi
                y = torch.tensor([[max(bearing,0), max(-bearing,0)]], dtype=torch.float)

                # Calculate loss
                loss = loss_fn(y_hat, y)
                losses[i,e] = loss.detach()

                # Calculate gradients
                grad = loss.backward()

                # Update weights & biases
                optimizer.step()

            if progress_bar and e % 1 == 0:
                printProgressBar(
                    iteration=e, total=epochs-1,
                    prefix='Progress', suffix=f'Loss: {sum(losses[:,e])/len(losses[:,e])}',
                    length=50
                )
        
        if loss_graph:
            x_vals = np.arange(losses.shape[1])
            for y_vals in losses:
                plt.plot(x_vals, y_vals)
            plt.show()


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

            Wss.append(nn.Parameter(torch.Tensor(Ws)))
            bss.append(nn.Parameter(torch.Tensor(bs)))

            start_idx = end_idx

        return Wss, bss
    
    @classmethod
    def get_wb_from_agent(cls, agent:"AgentNN") -> tuple[nn.ParameterList,nn.ParameterList]:
        return cls.get_wb_from_genotype(agent.genotype, agent.layers)