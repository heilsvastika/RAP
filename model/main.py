import os
import time
import argparse
import sys
import math
from sampler import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.backends import cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from util import *
import random
from critic import CriticNetwork
from mlp import MLP
from copy import deepcopy
import copy
import numpy as np
from collections import defaultdict

datatrain='ml-1m'
batch_size=128
lr=0.001
maxlen=5
hidden_size=50
embed_dim=100
num_epochs=200
dropout_rate=0.005
l2_emb=0.0
test='store_true'
topk=10
tau=0.1
seed=int(1000 * time.time())
samplecnt=5
epsilon=0.05
alpha=0.1


def Sample_RL(actor, critic, seq, x_embed, nxt, embed, Random=True):
    # current_lower_state=torch.zeros(1,2*self.hidden_size)
    nxt = torch.LongTensor([nxt])     # Target item
    current_lower_state = embed(nxt)  # get the embedding of target item  [1,embed_dim]

    # current_lower_state=current_lower_state.unsqueeze(0)   # [1,1,embed_dim]
    h = current_lower_state[0, 0:hidden_size].unsqueeze(0)  #[1,hidden_size]
    c = current_lower_state[0, hidden_size:].unsqueeze(0)   #[1,hidden_size]

    h=torch.unsqueeze(h,0)    # [1,1,hidden_size]
    c=torch.unsqueeze(c,0)    # [1,1,hidden_size]

    actions = []     # action list
    states = []      # state list
    seq_r = []       # sequence retained
    seq_d = []       # sequence deleted
    len_r = 0        # the length of sequence retained
    len_d = 0        # the length of sequence deleted
    prob_r = []      # the probability of sequence retained
    prob_d = []      # the probability of sequence deleted
    for pos in range(0, maxlen):
        # learn the action policy
        current_lower_state=torch.cat((h,c),2)  # [1,1,embed_dim]
        current_lower_state=current_lower_state.view(1, 1, embed_dim)     # [1,1,embed_dim]

        predicted = actor.get_target_output(current_lower_state, x_embed[pos], scope="target")
        states.append([current_lower_state, x_embed[pos]])
        if Random:
            if random.random() > epsilon:
                action = (0 if random.random() < float(predicted[0].item()) else 1)
            else:
                action = (1 if random.random() < float(predicted[0].item()) else 0)
        else:
            action = np.argmax(predicted).item()
        actions.append(action)
        if action == 1:
            out_d,(h,c) = critic.forward_lstm1(current_lower_state,seq[pos],"target")
            prob_r.append(float(predicted[0].item()))
        else:
            prob_d.append(float(predicted[0].item()))
    for (i, a) in enumerate(actions):
        if a == 1:
            seq_r.append(int(seq[i].item()))
        else:
            seq_d.append(int(seq[i].item()))
    len_r = len(seq_r)
    len_d = len(seq_d)
    if len(prob_r) == 0:
        prob_r.append(0.0)
    if len(prob_d) == 0:
        prob_d.append(0.0)

    if len_r == 0:
        actions[maxlen - 2] = 1
        seq_r.append(seq[maxlen - 2])
        len_r = 1
    if len_d == 0:
        actions[0] = 1
        seq_d.append(seq[0])
        len_d = 1
    # seq_r+=[1]*(self.maxlen-len_r)

    seq_r = torch.tensor(seq_r).view(1, -1)

    return actions, states, seq_r, len_r, seq_d, len_d, prob_r, prob_d

def Sample_pre(actor, critic, seq, x_embed, nxt, embed, Random=True):
    # current_lower_state=torch.zeros(1,2*self.hidden_size)
    nxt = torch.LongTensor([nxt])     # Target item

    current_lower_state = embed(nxt)  # get the embedding of target item  [1,embed_dim]
    print("current_lower_state:",current_lower_state.shape)

    #current_lower_state=current_lower_state.unsqueeze(0)   # [1,1,embed_dim]
    h = current_lower_state[0, 0:hidden_size].unsqueeze(0)  #[1,hidden_size]
    c = current_lower_state[0, hidden_size:].unsqueeze(0)   #[1,hidden_size]

    h=torch.unsqueeze(h,0)   # [1,1,hidden_size]
    c=torch.unsqueeze(c,0)   # [1,1,hidden_size]

    actions = []     # action list
    states = []      # state list
    seq_r = []       # sequence retained
    seq_d = []       # sequence deleted
    len_r = 0        # the length of sequence retained
    len_d = 0        # the length of sequence deleted
    prob_r = []      # the probability of sequence retained
    prob_d = []      # the probability of sequence deleted
    for pos in range(0, maxlen):
        # learn the action policy
        current_lower_state=torch.cat((h,c),2)
        current_lower_state=current_lower_state.view(1, 1, embed_dim)     # [1,1,embed_dim]
        print("current.size:",current_lower_state.shape)
        print("x_embed[",pos,"].size:",x_embed[pos].shape)
        predicted = actor.get_target_output(current_lower_state, x_embed[pos], scope="target")
        states.append([current_lower_state, x_embed[pos]])
        if Random:
            if random.random() > epsilon:
                action = (0 if random.random() < float(predicted[0].item()) else 1)
            else:
                action = (1 if random.random() < float(predicted[0].item()) else 0)
        else:
            action = np.argmax(predicted).item()
        actions.append(action)
        if action == 1:
            out_d,(h,c) = critic.forward_lstm1(current_lower_state,seq[pos],"target")
            prob_r.append(float(predicted[0].item()))
        else:
            prob_d.append(float(predicted[0].item()))
    for (i, a) in enumerate(actions):
        if a == 1:
            seq_r.append(int(seq[i].item()))
        else:
            seq_d.append(int(seq[i].item()))
    len_r = len(seq_r)
    len_d = len(seq_d)
    if len(prob_r) == 0:
        prob_r.append(0.0)
    if len(prob_d) == 0:
        prob_d.append(0.0)

    if len_r == 0:
        actions[maxlen - 2] = 1
        seq_r.append(seq[maxlen - 2])
        len_r = 1
    if len_d == 0:
        actions[0] = 1
        seq_d.append(seq[0])
        len_d = 1
    # seq_r+=[1]*(self.maxlen-len_r)

    seq_r = torch.tensor(seq_r).view(1, -1)

    return actions, seq_r, len_r, seq_d, len_d


class policyNet(nn.Module):
    def __init__(self):
        super(policyNet,self).__init__()
        self.hidden=hidden_size
        self.embed_dim=embed_dim
        self.W1=nn.Parameter(torch.FloatTensor(self.embed_dim,1).uniform_(-0.5,0.5))
        self.W2=nn.Parameter(torch.FloatTensor(self.embed_dim,1).uniform_(-0.5,0.5))
        self.b=nn.Parameter(torch.FloatTensor(1,1).uniform_(-0.5,0.5))

    def forward(self,h,x):
        #h0=torch.tensor([item.cpu().detach().numpy() for item in h])    #[1,1,embed_dim]
        h1=h[0]     # [1,embed_dim]
        x1=torch.squeeze(x,0)    # [1,embed_dim]

        h_ = torch.matmul(h1.view(1, -1), self.W1)  # 1x1
        x_ = torch.matmul(x1.view(1, -1), self.W2)  # 1x1

        scaled_out = torch.sigmoid(h_ + x_ + self.b)  # 1x1
        scaled_out = torch.clamp(scaled_out, min=1e-5, max=1 - 1e-5)
        scaled_out = torch.cat([1.0 - scaled_out, scaled_out], 0)
        return scaled_out


class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork,self).__init__()
        self.target_policy = policyNet()
        self.active_policy = policyNet()

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
            x.data = deepcopy(params[i].data * (tau) + x.data * (1-tau))
            i+=1

    def assign_active_network(self):
        params = []
        for name,x in self.target_policy.named_parameters():
            params.append(x)
        i = 0
        for name,x in self.active_policy.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1


def train_model(critic,actor,user_train,epoch, itemnum,sampler,embed,RL_train=True,LSTM_train=True):
    total_epoch_loss=0
    # get the batch number
    num_batch = int(math.floor(len(user_train)/batch_size))

    # critic network optimizer
    critic_target_optimizer1 = torch.optim.Adam(critic.target_pred1.parameters())
    critic_active_optimizer1 = torch.optim.Adam(critic.active_pred1.parameters())
    critic_target_optimizer2 = torch.optim.Adam(critic.target_pred2.parameters())
    critic_active_optimizer2 = torch.optim.Adam(critic.active_pred2.parameters())

    # actor network optimizer
    actor_target_optimizer = torch.optim.Adam(actor.target_policy.parameters())
    actor_active_optimizer = torch.optim.Adam(actor.active_policy.parameters())

    # two LSTM networks for two subsequences respectively
    #lstm1 = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=1,batch_first=True)
    #lstm2 = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=1,batch_first=True)

    # two MLPs
    classifier1 = MLP(hidden_size, itemnum)
    classifier2 = MLP(hidden_size, itemnum)
    # Softmax
    sg = nn.Sigmoid()
    # f = open(os.path.join('./data/obj.txt'), 'w')
    loss_fn = F.cross_entropy

    critic.assign_active_network()
    actor.assign_active_network()

    # process one batch of dataset
    for i in range(num_batch):
        u, seq, nxt = sampler.next_batch()   # user,sequence,target item
        u = np.array(u)     # user should plus one
        seq = np.array(seq)
        nxt = np.array(nxt)

        seq0 = seq     # initial sequence
        seq = torch.LongTensor([seq])  # (1,batch_size,maxlen)

        # print("seq0.view(1,-1):",seq0.view(1,-1))
        # define hidden states and cell states of subsequences through the LSTMs
        h_r0 = torch.zeros([1, 1, hidden_size])
        c_r0 = torch.zeros([1, 1, hidden_size])
        h_d0 = torch.zeros([1, 1, hidden_size])
        c_d0 = torch.zeros([1, 1, hidden_size])

        # embed the sequence into tensor
        x_embed = embed(seq)    # [1,batch_size,maxlen,embed_dim]

        critic.train()
        actor.train()
        critic_active_optimizer1.zero_grad()
        critic_target_optimizer1.zero_grad()
        critic_active_optimizer2.zero_grad()
        critic_target_optimizer2.zero_grad()

        r_sum = r = 0.
        # model training in one batch of dataset
        for j in range(0, batch_size):
            #print("batch[",j+1,"]")
            if RL_train:
                critic.train(False)
                actor.train()
                actionlist = []  # action list
                statelist = []  # state list
                losslist = []
                objlist = []
                avgloss = 0.0
                r_avg = 0.0
                r_tot = 0.0

                r=0.
                for k in range(samplecnt):
                    actions,states,seq_r,len_r,seq_d,len_d,prob_r,prob_d=Sample_RL(actor,critic,seq[0][j],
                                                                            x_embed[0][j], nxt[j], embed)
                    actionlist.append(actions)
                    statelist.append(states)

                    # Add the target item to the two subsequences split from sequence
                    seq_r0 = seq_r[0]      # subsequence retained
                    seq_d0 = seq_d         # subsequence deleted
                    seq_r1 = seq_r0.numpy().tolist()

                    # check the lengths of two subsequences
                    seq_r2 = torch.LongTensor([seq_r1])
                    seq_d2 = torch.LongTensor([seq_d0])
                    nxt1 = torch.LongTensor([nxt[j]])

                    seq_rt = embed(seq_r2)     # [1,len_r,embed_dim]
                    seq_dt = embed(seq_d2)     # [1,len_d,embed_dim]
                    nxt2 = embed(nxt1)    # the embedding of target item

                    output_r,h1_1,output_d, h2_1=critic.forward_lstm2(h_r0, c_r0, seq_rt, h_d0, c_d0, seq_dt,"target")

                    #output_r, (h_rn, c_rn) = lstm1(seq_rt, (h_r0, c_r0))  # output:(128,maxlen,50), hn-cn:(1,maxlen,50)
                    #output_d, (h_dn, c_dn) = lstm2(seq_dt, (h_d0, c_d0))

                    output1_0 = h1_1[0]    # the output of positive subsequence through the LSTM
                    output2_0 = h2_1[0]    # the output of negative subsequence through the LSTM

                    mlp1=MLP(hidden_size,embed_dim)
                    mlp2=MLP(hidden_size,embed_dim)
                    output1 = mlp1.forward(output1_0)
                    output2 = mlp2.forward(output2_0)

                    reward1 = torch.cosine_similarity(output1,nxt2,dim=1)
                    reward2 = torch.cosine_similarity(output2,nxt2,dim=1)

                    # calculate the probability of subsequences
                    p1 = 1.0
                    p2 = 1.0
                    for _, prob_r1 in enumerate(prob_r):
                        p1 *= prob_r1
                    for _, prob_d1 in enumerate(prob_d):
                        p2 *= (1 - prob_d1)

                    # calculate the reward
                    r1 = p1 * reward1     # reward for positive subsequence
                    r2 = p2 * reward2     # reward for negative subsequence
                    r = r2 - r1     # 相当于loss
                    r += (float(len_r)/maxlen)**2 *0.15
                    r_avg += r
                    losslist.append(r)
                r_avg /= samplecnt
                r_tot += r_avg
                grad1=None
                grad2=None
                grad3=None
                flag=0
                if LSTM_train:
                    #print("RL and LSTM True!")
                    critic.train()
                    actor.train()
                    critic_active_optimizer1.zero_grad()
                    critic_active_optimizer2.zero_grad()
                    critic_target_optimizer1.zero_grad()
                    critic_target_optimizer2.zero_grad()

                    output_r, h1_1, output_d, h2_1 = critic.forward_lstm2(h_r0, c_r0, seq_rt, h_d0, c_d0, seq_dt,
                                                                          "target")
                    output1_0 = h1_1[0]  # the output of positive subsequence through the LSTM
                    output2_0 = h2_1[0]  # the output of negative subsequence through the LSTM

                    mlp1 = MLP(hidden_size, embed_dim)
                    mlp2 = MLP(hidden_size, embed_dim)
                    output1 = mlp1.forward(output1_0)
                    output2 = mlp2.forward(output2_0)

                    reward1 = torch.cosine_similarity(output1, nxt2, dim=1)
                    reward2 = torch.cosine_similarity(output2, nxt2, dim=1)
                    # calculate the probability of subsequences
                    p1 = 1.0
                    p2 = 1.0
                    for _, prob_r1 in enumerate(prob_r):
                        p1 *= prob_r1
                    for _, prob_d1 in enumerate(prob_d):
                        p2 *= (1 - prob_d1)

                    # calculate the reward
                    r1 = p1 * reward1  # reward for positive subsequence
                    r2 = p2 * reward2  # reward for negative subsequence
                    r = r2 - r1  # 相当于loss

                    r.backward()

                    critic.assign_active_network_gradients()
                    critic_active_optimizer1.step()
                    critic_target_optimizer2.step()

                for i in range(samplecnt):
                    for pos in range(len(actionlist)):
                        rr=[0,0]
                        rr[actionlist[i][pos]]=((losslist[i]-r_avg)*alpha).item()
                        g=actor.get_gradient(statelist[i][pos][0],statelist[i][pos][1],rr,scope="target")
                        if flag==0:
                            grad1=g[0]
                            grad2=g[1]
                            grad3=g[2]
                            flag=1
                        else:
                            grad1 += g[0]
                            grad2 += g[1]
                            grad3 += g[2]
                        #print("grad3:",grad3)
                actor_target_optimizer.zero_grad()
                actor_active_optimizer.zero_grad()
                actor.assign_active_network_gradients(grad1,grad2,grad3)
                actor_active_optimizer.step()
            else:
                #print("RL False LSTM True")
                critic.train()
                actor.train(False)
                critic_active_optimizer1.zero_grad()
                critic_active_optimizer2.zero_grad()
                critic_target_optimizer1.zero_grad()
                critic_target_optimizer2.zero_grad()

                output_r, h1_1, output_d, h2_1 = critic.forward_lstm2(h_r0, c_r0, seq_rt, h_d0, c_d0, seq_dt,
                                                                      "target")
                output1_0 = h1_1[0]  # the output of positive subsequence through the LSTM
                output2_0 = h2_1[0]  # the output of negative subsequence through the LSTM

                mlp1 = MLP(hidden_size, embed_dim)
                mlp2 = MLP(hidden_size, embed_dim)
                output1 = mlp1.forward(output1_0)
                output2 = mlp2.forward(output2_0)

                reward1 = torch.cosine_similarity(output1, nxt2, dim=1)
                reward2 = torch.cosine_similarity(output2, nxt2, dim=1)
                # calculate the probability of subsequences
                p1 = 1.0
                p2 = 1.0
                for _, prob_r1 in enumerate(prob_r):
                    p1 *= prob_r1
                for _, prob_d1 in enumerate(prob_d):
                    p2 *= (1 - prob_d1)

                # calculate the reward
                r1 = p1 * reward1  # reward for positive subsequence
                r2 = p2 * reward2  # reward for negative subsequence
                r = r2 - r1  # 相当于loss
                r_avg+=r.item()
                r.backward()
                critic.assign_active_network_gradients()
                critic_active_optimizer1.step()
                critic_active_optimizer2.step()

        if RL_train:
            #print("Again RL True!")
            critic.train(False)
            actor.train()
            actor.update_target_network()
            if LSTM_train:
                critic.train()
                actor.train()
                critic.update_target_network()
        else:
            critic.train()
            actor.train(False)
            critic.assign_target_network()
        r_avg /=batch_size
        total_epoch_loss+=r_avg
    return total_epoch_loss/len(user_train)

    #_,prediction=critic.forward_lstm1(hid,seq_r2,scope="target")
    #a = torch.FloatTensor(1, itemnum).zero_()
    #b = a.scatter(dim=1, index=torch.LongTensor([[nxt[j] - 1]]), value=1)


def main():
    random.seed(seed)

    dataset = data_partition(datatrain)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print("usernum:",usernum)
    print("itemnum:",itemnum)
    num_batch = int(math.floor(len(user_train) / batch_size))
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f=open(os.path.join('log.txt'),'w')

    embed = nn.Embedding(itemnum,embed_dim)
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=batch_size, maxlen=maxlen, n_workers=3)
    #sampler_valid=WarpSampler(user_valid,usernum,itemnum,batch_size=batch_size,maxlen=maxlen,n_workers=3)

    critic = CriticNetwork(batch_size,hidden_size,itemnum,embed_dim,embed,tau)
    actor = ActorNetwork()

    T=0.0
    t0=time.time()
    for epoch in range(num_epochs):
        loss=train_model(critic, actor, user_train, epoch, itemnum, sampler, embed)
        print("epoch-",epoch+1,":")
        print("epoch-loss:",loss.item())
        f.write('epoch-'+str(epoch+1)+'-loss:'+str(loss.item())+'\n')
        f.flush()
        if epoch >= 0:
            t1=time.time()-t0
            T+=t1
            # print("Evaluating...")
            

    f.close()

if __name__=='__main__':
    main()
