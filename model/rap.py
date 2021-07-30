import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from actor import ActorNetwork
from critic import CriticNetwork
import random
import numpy as np
from mlp import MLP
import os

class RAP(nn.Module):
    """
    Args:
        n_items(int):the number of items
        hidden_size(int):the hidden size of gru
        embedding_dim(int):the dimension of item embedding
        batch_size(int):the size of batch
        n_layers(int):the number of gru layers
    """

    def __init__(self,n_items,args):
        super(RAP,self).__init__()
        self.n_items=n_items
        self.args=args
        self.hidden_size=args.hidden_size
        self.embed_dim=args.embed_dim
        self.batch_size=args.batch_size
        self.maxlen=args.maxlen
        self.embed = nn.Embedding(n_items, args.embed_dim)
        self.dropout=args.dropout_rate
        self.epsilon=0.05
        self.samplecnt=5
        self.batch_size0=1
        self.alpha=0.1
        self.actor=ActorNetwork(self.args)
        self.critic=CriticNetwork(self.args,2,self.n_items,self.embed)

        self.critic_target_optimizer = torch.optim.Adam(self.critic.target_pred.parameters())
        self.critic_active_optimizer = torch.optim.Adam(self.critic.active_pred.parameters())

        self.actor_target_optimizer = torch.optim.Adam(self.actor.target_policy.parameters())
        self.actor_active_optimizer = torch.optim.Adam(self.actor.active_policy.parameters())

        #LSTM
        self.lstm1 = nn.LSTM(input_size=self.embed_dim,hidden_size=args.hidden_size,num_layers=1,batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.embed_dim,hidden_size=args.hidden_size,num_layers=1,batch_first=True)

        # MLP
        self.classifier1=MLP(self.hidden_size,self.n_items)
        self.classifier2=MLP(self.hidden_size,self.n_items)

        # Softmax
        self.sg=nn.Sigmoid()
        self.f = open(os.path.join('./data/obj.txt'), 'w')
        self.loss_fn=F.cross_entropy


    def forward(self,seq,nxt,hidden=None):
        #print("seq size:",seq.shape)   # seq:(batch_size,maxlen)
        seq0=seq
        seq=torch.LongTensor([seq])   # seq:(1,batch_size,maxlen)

        #print("seq0.view(1,-1):",seq0.view(1,-1))
        h_r0 = torch.zeros([1, self.batch_size0, self.hidden_size])
        c_r0 = torch.zeros([1, self.batch_size0, self.hidden_size])

        h_d0 = torch.zeros([1, self.batch_size0, self.hidden_size])
        c_d0 = torch.zeros([1, self.batch_size0, self.hidden_size])

        # embed the sequence into tensor
        x_embed = self.embed(seq)   # x_embed:(1,batch_size,maxlen,hidden_size),x_embed[0][i]:(maxlen,embed_dim)
        #nxt_embed = self.embed(nxt)   # nxt_embed: (1,batch_size,embed_dim)

        #LSTM
        #output, (hn, cn) = self.lstm(x_embed[0], (h0, c0))  # output:(128,maxlen,50), hn-cn:(1,maxlen,50)

        self.critic.train()
        self.actor.train()
        self.critic_active_optimizer.zero_grad()
        self.critic_target_optimizer.zero_grad()
        #print("Done!")

        actionlist=[]
        statelist=[]
        losslist=[]
        objlist=[]
        avgloss=0.0
        aveloss=0.0
        totloss=0.0

        for i in range(0,self.batch_size):
            for j in range(self.samplecnt):
                actions,states,seq_r,len_r,seq_d,len_d,prob_r,prob_d=self.Sample_RL(self.actor,seq[0][i],x_embed[0][i],nxt[i],self.maxlen,self.epsilon)
                #print("actions:",actions)
                #print("seq:",seq[0][i])
                #print("nxt:",nxt[i])
                actionlist.append(actions)
                statelist.append(states)


                # Add the target item to the two subsequences split from sequence
                seq_r0=seq_r[0]
                seq_d0=seq_d
                seq_r1=seq_r0.numpy().tolist()
                seq_r1.append(nxt[i])
                seq_d0.append(nxt[i])

                seq_r2=torch.LongTensor([seq_r1])
                seq_d2=torch.LongTensor([seq_d0])
                seq_rt=self.embed(seq_r2)
                seq_dt=self.embed(seq_d2)

                output_r, (h_rn, c_rn) = self.lstm1(seq_rt, (h_r0, c_r0))  # output:(128,maxlen,50), hn-cn:(1,maxlen,50)
                output_d, (h_dn, c_dn) = self.lstm2(seq_dt, (h_d0, c_d0))

                output1_0=self.classifier1.forward(h_rn[0])
                output2_0=self.classifier2.forward(h_dn[0])

                output1=self.sg(output1_0[0])
                output2=self.sg(output2_0[0])
                #outputx=nn.functional.softmax(output1_0[0],0)
                #outputy=nn.functional.softmax(output2_0[0],0)


                #sum=torch.sum(output1)     #get the sum of elements in tensor
                a = torch.FloatTensor(1, 3417).zero_()
                b = a.scatter(dim=1, index=torch.LongTensor([[nxt[i]-1]]), value=1)
                y=nxt[i]-1
                print("output1:",output1)
                print("y:",y)
                print("output1[y]:",output1[y].item())
                loss=-torch.log(output1[y])
                print("cross_entropy:",loss)
                aveloss+=loss
                losslist.append(loss)


                #output2=self.sg(output2_0[0])

                # get the output of MLP and obtain the golden probability
                xyz1=output1[nxt[i]-1]
                xyz2=output2[nxt[i]-1]

                reward1_0=xyz1
                reward2_0=xyz2
                reward1=reward1_0.item()
                reward2=reward2_0.item()
                print("reward1:",reward1,"reward2:",reward2)
                print("reward:",reward1-reward2)
                p1=1.0
                p2=1.0
                for _,prob_r1 in enumerate(prob_r):
                    p1*=prob_r1
                for _,prob_d1 in enumerate(prob_d):
                    p2*=(1-prob_d1)

                # objective definition
                objective=0.0
                objective=p1*reward1-p2*reward2
                print("objective:",objective)

                # output the objective to file
                #self.f.write(str(i)+"-objective:"+str(objective)+"\n")
                self.critic.assign_active_network_gradients()
                self.critic_active_optimizer.step()

            aveloss/=self.samplecnt
            totloss+=aveloss
            grad1=None
            grad2=None
            grad3=None
            flag=0

            for i in range(self.samplecnt):
                for pos in range(len(actionlist[i])):
                    rr=[0,0]
                    rr[actionlist[i][pos]]=((losslist[i]-aveloss)*self.alpha).item()

                    g=self.actor.get_gradient(statelist[i][pos][0],statelist[i][pos][1],rr,scope="target")

                    if flag==0:
                        grad1=g[0]
                        grad2=g[1]
                        grad3=g[2]
                        flag=1
                    else:
                        grad1+=g[0]
                        grad2+=g[1]
                        grad3+=g[2]
            self.actor_target_optimizer.zero_grad()
            self.actor_active_optimizer.zero_grad()
            self.actor.assign_active_network_gradients(grad1,grad2,grad3)
            self.actor_active_optimizer.step()
            #print("after:active: ",self.actor.active_policy.b,"target: ",self.actor.target_policy.b)

        self.critic.train(False)
        self.actor.train()
        self.actor.update_target_network()



    def Sample_RL(self,actor,seq,x_embed,nxt,length,epsilon,Random=True):
        #current_lower_state=torch.zeros(1,2*self.hidden_size)
        nxt = torch.LongTensor([nxt])
        current_lower_state=self.embed(nxt)

        #print("current_lower_state.size:",current_lower_state.size())
        actions=[]
        states=[]
        seq_r=[]
        seq_d=[]
        len_r=0
        prob_r=[]
        prob_d=[]
        for pos in range(0,self.maxlen):
            predicted=actor.get_target_output(current_lower_state,x_embed[pos],scope="target")
            states.append([current_lower_state,x_embed[pos]])
            if Random:
                if random.random()>epsilon:
                    action=(0 if random.random() < float(predicted[0].item()) else 1)
                else:
                    action=(1 if random.random() < float(predicted[0].item()) else 0)
            else:
                action=np.argmax(predicted).item()
            actions.append(action)
            if action==1:
                out_d,current_lower_state=self.critic.forward_lstm(current_lower_state,seq[pos],"target")
                prob_r.append(float(predicted[0].item()))
            else:
                prob_d.append(float(predicted[0].item()))
        for (i,a) in enumerate(actions):
            if a==1:
                seq_r.append(int(seq[i].item()))
            else:
                seq_d.append(int(seq[i].item()))
        len_r=len(seq_r)
        len_d=len(seq_d)
        if len(prob_r)==0:
            prob_r.append(0.0)
        if len(prob_d)==0:
            prob_d.append(0.0)

        if len_r==0:
            actions[self.maxlen-2]=1
            seq_r.append(seq[self.maxlen-2])
            len_r=1
        #seq_r+=[1]*(self.maxlen-len_r)

        seq_r=torch.tensor(seq_r).view(1,-1)

        return actions,states,seq_r,len_r,seq_d,len_d,prob_r,prob_d
