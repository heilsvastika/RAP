import torch
import torch.nn as nn
import torch.nn.functional as F

# define a multi-layer neural network
class MLP(nn.Module):
    def __init__(self,input_size,common_size):
        super(MLP,self).__init__()
        self.fc1=nn.Linear(input_size,input_size*2)
        self.fc2=nn.Linear(input_size*2,input_size*2)
        self.fc3=nn.Linear(input_size*2,common_size)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
