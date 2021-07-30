import torch.nn as nn
import torch

# define a action policy network
class policyNet(nn.Module):
    def __init__(self,args):
        super(policyNet,self).__init__()
        self.hidden=args.hidden_size
        self.embed_dim=args.embed_dim
        self.W1=nn.Parameter(torch.FloatTensor(2*self.hidden,1).uniform_(-0.5,0.5))
        self.W2=nn.Parameter(torch.FloatTensor(self.embed_dim,1).uniform_(-0.5,0.5))
        self.b=nn.Parameter(torch.FloatTensor(1,1).uniform_(-0.5,0.5))

    def forward(self,h,x):
        h_=torch.matmul(h.view(1,-1),self.W1)    # 1*1
        x_=torch.matmul(x.view(1,-1),self.W2)    # 1*1
        scaled_out=torch.sigmoid(h_+x_+self.b)   # 1*1
        scaled_out=torch.cat([1.0-scaled_out,scaled_out],0)
        return scaled_out
