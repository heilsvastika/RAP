import torch
import torch.nn as nn
from policynet import policyNet
from copy import deepcopy

class ActorNetwork(nn.Module):
"""
    action network
    use the state
    sample the action
"""
    def __init__(self,args):
        super(ActorNetwork,self).__init__()
        self.target_policy = policyNet(args)
        self.active_policy = policyNet(args)
        self.tau = args.tau

    def get_target_output(self,h,x,scope):
        if scope == "target":
            out = self.target_policy(h,x)
        if scope == "active":
            out = self.active_policy(h,x)
        return out

    def get_target_logOutput(self,h,x):
        out = self.target_policy(h,x)
        logOut = torch.log(out)
        return logOut

    def get_gradient(self,h,x,reward,scope):
        if scope == "target":
            out = self.target_policy(h,x)
            logout = torch.log(out).view(-1)
            index = reward.index(0)
            index = (index+1)%2
            #print(out,reward,index,logout[index].view(-1),logout)
            #print(logout[index].view(-1)
            grad = torch.autograd.grad(logout[index].view(-1),self.target_policy.parameters())
            #print(grad[0].size(),grad[1].size(),grad[2].size())
            #print(grad[0],grad[1],grad[2])
            grad[0].data = grad[0].data * reward[index]
            grad[1].data = grad[1].data * reward[index]
            grad[2].data = grad[2].data * reward[index]
            #print(grad[0],grad[1],grad[2])
            return grad
        if scope == "active":
            out = self.active_policy(h,x)
        return out

    def assign_active_network_gradients(self,grad1,grad2,grad3):
        params = [grad1,grad2,grad3]
        i=0
        for name,x in self.active_policy.named_parameters():
            x.grad = deepcopy(params[i])
            i+= 1

    def update_target_network(self):
        params = []
        for name,x in self.active_policy.named_parameters():
            params.append(x)
        i = 0
        for name,x in self.target_policy.named_parameters():
            x.data = deepcopy(params[i].data * (self.tau) + x.data * (1-self.tau))
            i+=1

    def assign_active_network(self):
        params = []
        for name,x in self.target_policy.named_parameters():
            params.append(x)
        i = 0
        for name,x in self.active_policy.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1
