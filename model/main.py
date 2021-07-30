import os
import time
import argparse
import sys
from parser1 import Parser1
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
from rap import RAP
import random
from critic import CriticNetwork
from actor import ActorNetwork
from mlp import MLP


def Sample_RL(args, actor, critic, seq, x_embed, nxt, embed, Random=True):
    # current_lower_state=torch.zeros(1,2*self.hidden_size)
    nxt = torch.LongTensor([nxt])     # Target item
    current_lower_state = embed(nxt)  # get the embedding of target item

    # print("current_lower_state.size:",current_lower_state.size())
    actions = []     # action list
    states = []      # state list
    seq_r = []       # sequence retained
    seq_d = []       # sequence deleted
    len_r = 0        # the length of sequence retained
    len_d = 0        # the length of sequence deleted
    prob_r = []      # the probability of sequence retained
    prob_d = []      # the probability of sequence deleted
    for pos in range(0, args.maxlen):
        # learn the action policy
        predicted = actor.get_target_output(current_lower_state, x_embed[pos], scope="target")
        states.append([current_lower_state, x_embed[pos]])
        if Random:
            if random.random() > args.epsilon:
                action = (0 if random.random() < float(predicted[0].item()) else 1)
            else:
                action = (1 if random.random() < float(predicted[0].item()) else 0)
        else:
            action = np.argmax(predicted).item()
        actions.append(action)
        if action == 1:
            out_d, current_lower_state = critic.forward_lstm(current_lower_state, seq[pos], "target")
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
        actions[args.maxlen - 2] = 1
        seq_r.append(seq[args.maxlen - 2])
        len_r = 1
    if len_d == 0:
        actions[0] = 1
        seq_d.append(seq[0])
        len_d = 1
    # seq_r+=[1]*(self.maxlen-len_r)

    seq_r = torch.tensor(seq_r).view(1, -1)

    return actions, states, seq_r, len_r, seq_d, len_d, prob_r, prob_d


def train_model(actor,critic,user_train,itemnum,sampler,args,embed,RL_train=True,LSTM_train=True):
    # get the batch number
    num_batch = int(math.floor(len(user_train)/args.batch_size))

    # critic network optimizer
    critic_target_optimizer = torch.optim.Adam(critic.target_pred.parameters())
    critic_active_optimizer = torch.optim.Adam(critic.active_pred.parameters())

    # actor network optimizer
    actor_target_optimizer = torch.optim.Adam(actor.target_policy.parameters())
    actor_active_optimizer = torch.optim.Adam(actor.active_policy.parameters())

    # two LSTM networks for two subsequences respectively
    lstm1 = nn.LSTM(input_size=args.embed_dim, hidden_size=args.hidden_size, num_layers=1,
                    batch_first=True)
    lstm2 = nn.LSTM(input_size=args.embed_dim, hidden_size=args.hidden_size, num_layers=1,
                    batch_first=True)
    # two MLPs
    classifier1 = MLP(args.hidden_size, itemnum)
    classifier2 = MLP(args.hidden_size, itemnum)
    # Softmax
    sg = nn.Sigmoid()
    # f = open(os.path.join('./data/obj.txt'), 'w')
    loss_fn = F.cross_entropy

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
        h_r0 = torch.zeros([1, 1, args.hidden_size])
        c_r0 = torch.zeros([1, 1, args.hidden_size])
        h_d0 = torch.zeros([1, 1, args.hidden_size])
        c_d0 = torch.zeros([1, 1, args.hidden_size])

        # embed the sequence into tensor
        x_embed = embed(seq)

        critic.train()
        actor.train()
        critic_active_optimizer.zero_grad()
        critic_target_optimizer.zero_grad()

        actionlist = []     # action list
        statelist = []      # state list
        losslist = []
        objlist = []
        avgloss = 0.0
        aveloss = 0.0
        totloss = 0.0

        r_sum=0.
        # model training in one batch of dataset
        for j in range(0, args.batch_size):
            r = 0.
            for k in range(args.samplecnt):
                actions,states,seq_r,len_r,seq_d,len_d,prob_r,prob_d=Sample_RL(args,actor,critic,seq[0][j],
                                                                            x_embed[0][j], nxt[j], embed)
                #print("actions:",actions)
                # print("seq:",seq[0][i])
                # print("nxt:",nxt[i])

                actionlist.append(actions)
                statelist.append(states)

                # Add the target item to the two subsequences split from sequence
                seq_r0 = seq_r[0]      # subsequence retained
                seq_d0 = seq_d         # subsequence deleted
                seq_r1 = seq_r0.numpy().tolist()
                # check the lengths of two subsequences
                #print("the length of seq_r1:",len(seq_r1))
                #print("the length of seq_d0:",len(seq_d))

                seq_r2 = torch.LongTensor([seq_r1])
                seq_d2 = torch.LongTensor([seq_d0])
                nxt1 = torch.LongTensor([nxt[j]])
                seq_rt = embed(seq_r2)
                seq_dt = embed(seq_d2)
                nxt2 = embed(nxt1)    # the embedding of target item

                output_r, (h_rn, c_rn) = lstm1(seq_rt, (h_r0, c_r0))  # output:(128,maxlen,50), hn-cn:(1,maxlen,50)
                output_d, (h_dn, c_dn) = lstm2(seq_dt, (h_d0, c_d0))

                output1_0 = h_rn[0]    # the output of positive subsequence through the LSTM
                output2_0 = h_dn[0]    # the output of negative subsequence through the LSTM

                mlp1=MLP(args.hidden_size,args.embed_dim)
                mlp2=MLP(args.hidden_size,args.embed_dim)
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
                reward = r1 - r2
                r += reward
            r_avg = r/(k+1)
            r_sum += r_avg
            r_sum_avg = r_sum/(args.batch_size)     # reward for one batch
        print("r_sum_avg:",r_sum_avg.item())

                #a = torch.FloatTensor(1, itemnum).zero_()
                #b = a.scatter(dim=1, index=torch.LongTensor([[nxt[j] - 1]]), value=1)
                #y = nxt[j] - 1




def main():
    # get parser
    argv = sys.argv[1:]
    parser = Parser1().getParser()
    args, _ = parser.parse_known_args(argv)
    random.seed(args.seed)

    dataset = data_partition(args.datatrain)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    num_batch = int(math.floor(len(user_train) / args.batch_size))
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    embed = nn.Embedding(itemnum, args.embed_dim)
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    critic=CriticNetwork(args,itemnum,embed)
    actor=ActorNetwork(args)
    for epoch in range(args.num_epochs):
        train_model(actor,critic,user_train,itemnum,sampler,args,embed)



if __name__=='__main__':
    main()
