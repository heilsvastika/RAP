import torch
import torch.nn as nn
from lstm import LSTMClassifier
from copy import deepcopy

class CriticNetwork(nn.Module):
    """
    predict network
    use the word vector and actions(sampled from actor network)
    get the final prediction
    """
    def __init__(self,args,itemnum,embed):
        super(CriticNetwork,self).__init__()
        self.target_pred = LSTMClassifier(args,itemnum,embed)
        self.active_pred = LSTMClassifier(args,itemnum,embed)
        self.tau = args.tau

    def forward(self,x,scope):
        if scope == "target":
            out = self.target_pred(x)
        if scope == "active":
            out = self.active_pred(x)
        return out

    def assign_target_network(self):
        params = []
        for name,x in self.active_pred.named_parameters():
            params.append(x)
        i=0
        for name,x in self.target_pred.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1

    def update_target_network(self):
        params = []
        for name,x in self.active_pred.named_parameters():
            params.append(x)
        i=0
        for name,x in self.target_pred.named_parameters():
            x.data = deepcopy(params[i].data*(self.tau)+x.data * (1-self.tau))
            i+=1

    def assign_active_network(self):
        params = []
        for name,x in self.target_pred.named_parameters():
            params.append(x)
        i=0
        for name,x in self.active_pred.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1

    def assign_active_network_gradients(self):
        params = []
        for name,x in self.target_pred.named_parameters():
            params.append(x)
        i=0
        for name,x in self.active_pred.named_parameters():
            x.grad = deepcopy(params[i].grad)
            i+=1
        for name,x in self.target_pred.named_parameters():
            x.grad = None

    def forward_lstm(self,hc,x,scope):
        if scope == "target":
            out,state = self.target_pred.getNextHiddenState(hc,x)
        if scope == "active":
            out,state = self.active_pred.getNextHiddenState(hc,x)
        return out,state

    def ivector_find(self,x):
        return self.target_pred.ivector_find(x)
