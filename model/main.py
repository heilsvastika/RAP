import math
import os
import time
from copy import deepcopy
import torch.nn.functional as F
from torch import nn
from critic import CriticNetwork
from mlp import MLP
from sampler import *
from util import *

data_file='ml-1m'
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


def normalize(inputs,epsilon):
    inputs_shape=inputs.get_shape()
    params_shape=inputs_shape[-1:]


def Sample_RL(actor, critic, seq, x_embed, nxt, embed, Random=True):
    # current_lower_state=torch.zeros(1,2*self.hidden_size)
    # nxt = torch.LongTensor(nxt)     # Target item
    current_lower_state = torch.unsqueeze(nxt,0)  # get the embedding of target item  [1,embed_dim]

    # current_lower_state=current_lower_state.unsqueeze(0)   # [1,1,embed_dim]
    h_0 = current_lower_state[0, 0:hidden_size]  #[1,hidden_size]
    c_0 = current_lower_state[0, hidden_size:]   #[1,hidden_size]

    h_1=torch.unsqueeze(h_0,0)
    c_1=torch.unsqueeze(c_0,0)

    h=torch.unsqueeze(h_1,0)    # [1,1,hidden_size]
    c=torch.unsqueeze(c_1,0)    # [1,1,hidden_size]

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
            seq_r.append(seq[i])
        else:
            seq_d.append(seq[i])
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

    return actions, states, seq_r, len_r, seq_d, len_d, prob_r, prob_d

def Sample_pre(actor, critic, seq, x_embed, nxt, embed, Random=True):
    # current_lower_state=torch.zeros(1,2*self.hidden_size)
    current_lower_state = nxt[0]  # get the embedding of target item  [1,embed_dim]

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

    #seq_emb=embed(x_embed)   # [1,maxlen,embed_dim]
    seq_emb=x_embed[0]   # [maxlen,embed_dim]

    for pos in range(0, maxlen):
        # learn the action policy
        current_lower_state=torch.cat((h,c),2)   # [1,1,embed_dim]
        current_lower_state=current_lower_state.view(1,1,embed_dim)     # [1,1,embed_dim]

        seq1=torch.unsqueeze(seq_emb[pos],0)
        seq2=torch.unsqueeze(seq1,0)   # [1,1,embed_dim]
        x_embed_1=x_embed[0]    # [maxlen,embed_dim]
        x_embed_2=torch.unsqueeze(x_embed_1[pos],0)    # [1,embed_dim]
        x_embed1=torch.unsqueeze(x_embed_2,0)    # [1,1,embed_dim]

        predicted = actor.get_target_output(current_lower_state, x_embed1, scope="target")
        states.append([current_lower_state, x_embed1])
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
            seq_r.append(seq[i])
        else:
            seq_d.append(seq[i])
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

    # seq_r = torch.LongTensor(seq_r).view(1, -1)

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
        #h0=torch.LongTensor([item.cpu().detach().numpy() for item in h])    #[1,1,embed_dim]
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
        seq = torch.LongTensor([seq])  # [1,batch_size,maxlen]
        nxt = torch.LongTensor([nxt])  # [1,batch_size]

        # print("seq0.view(1,-1):",seq0.view(1,-1))
        # define hidden states and cell states of subsequences through the LSTMs
        h_r0 = torch.zeros([1, 1, hidden_size])
        c_r0 = torch.zeros([1, 1, hidden_size])
        h_d0 = torch.zeros([1, 1, hidden_size])
        c_d0 = torch.zeros([1, 1, hidden_size])

        # embed the sequence into tensor
        x_embed = embed(seq)    # [1,batch_size,maxlen,embed_dim]
        nxt_embed = embed(nxt)  # [1,batch_size,embed_size]

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
                avgloss = 0.0
                r_avg = 0.0
                r_tot = 0.0

                r=0.0
                for k in range(samplecnt):
                    actions,states,seq_r,len_r,seq_d,len_d,prob_r,prob_d=Sample_RL(actor,critic,seq0[j],
                                                                            x_embed[0][j], nxt_embed[0][j], embed)
                    actionlist.append(actions)
                    statelist.append(states)

                    # Add the target item to the two subsequences split from sequence
                    seq_r0 = seq_r      # subsequence retained
                    seq_d0 = seq_d      # subsequence deleted
                    #seq_r1 = seq_r0.numpy().tolist()

                    # check the lengths of two subsequences
                    seq_r1 = torch.LongTensor(seq_r0)   # [len_r]
                    seq_d1 = torch.LongTensor(seq_d0)   # [len_d]
                    nxt1 = nxt_embed[0][j]  # [100]

                    seq_rt = torch.unsqueeze(embed(seq_r1),0)     # [len_r,embed_dim]
                    seq_dt = torch.unsqueeze(embed(seq_d1),0)    # [len_d,embed_dim]

                    output_r,h1_1,output_d,h2_1=critic.forward_lstm2(h_r0, c_r0, seq_rt, h_d0, c_d0, seq_dt,"target")

                    #output_r, (h_rn, c_rn) = lstm1(seq_rt, (h_r0, c_r0))  # output:(128,maxlen,50), hn-cn:(1,maxlen,50)
                    #output_d, (h_dn, c_dn) = lstm2(seq_dt, (h_d0, c_d0))

                    output1 = h1_1[0]    # the output of positive subsequence through the LSTM  [1,hidden_size]
                    output2 = h2_1[0]    # the output of negative subsequence through the LSTM  [1,hidden_size]

                    mlp=MLP(embed_dim,hidden_size)
                    nxt=mlp.forward(nxt1)    #[hidden_size]

                    reward1 = torch.cosine_similarity(output1[0],nxt,dim=0)
                    reward2 = torch.cosine_similarity(output2[0],nxt,dim=0)

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
                r_avg /= samplecnt    # average reward feedback
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
                    output1 = h1_1[0]  # the output of positive subsequence through the LSTM
                    output2 = h2_1[0]  # the output of negative subsequence through the LSTM

                    mlp = MLP(embed_dim,hidden_size)
                    nxt = mlp.forward(nxt_embed[0][j])

                    reward1 = torch.cosine_similarity(output1[0], nxt[0], dim=0)
                    reward2 = torch.cosine_similarity(output2[0], nxt[0], dim=0)

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
                    r = (r2 - r1)  # 相当于loss
                    r.backward(retain_graph=True)  # success

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
                output1 = h1_1[0]  # the output of positive subsequence through the LSTM
                output2 = h2_1[0]  # the output of negative subsequence through the LSTM

                mlp = MLP(embed_dim,hidden_size)
                nxt = mlp.forward(nxt_embed[0])  # 还有问题

                reward1 = torch.cosine_similarity(output1[0], nxt[0], dim=0)
                reward2 = torch.cosine_similarity(output2[0], nxt[0], dim=0)

                r1_0=reward1.item()
                r2_0=reward2.item()

                # calculate the probability of subsequences
                p1 = 1.0
                p2 = 1.0
                for _, prob_r1 in enumerate(prob_r):
                    p1 *= prob_r1
                for _, prob_d1 in enumerate(prob_d):
                    p2 *= (1 - prob_d1)

                # calculate the reward
                r1 = p1 * r1_0  # reward for positive subsequence
                r2 = p2 * r2_0  # reward for negative subsequence
                r = r2 - r1  # 相当于loss
                r_avg+=r
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
    #a = torch.LongTensor(1, itemnum).zero_()
    #b = a.scatter(dim=1, index=torch.LongTensor([[nxt[j] - 1]]), value=1)

def predict(actor,critic,u,input_seq,item_idx,embed):
    u = np.array(u)  # user should plus one
    seq_0 = np.array(input_seq)
    nxt_0 = np.array(item_idx)
    # print("pre-max:",nxt_0.max())
    # print("pre-min:",nxt_0.min())

    seq = seq_0  # initial sequence
    nxt = nxt_0
    seq = torch.LongTensor([seq])  # [1,maxlen]
    nxt = torch.LongTensor([nxt])  # [1,length]
    nxt1= embed(nxt)   # [1,length,embed_dim]

    # print("embedding.num_embeddings:",embed.num_embeddings)

    ndcg=0.0
    hit=0.0

    # define hidden states and cell states of subsequences through the LSTMs
    h_r0 = torch.zeros([1, 1, hidden_size])
    c_r0 = torch.zeros([1, 1, hidden_size])
    h_d0 = torch.zeros([1, 1, hidden_size])
    c_d0 = torch.zeros([1, 1, hidden_size])

    rewardlist=defaultdict(list)

    # model training in one batch of dataset
    length=len(item_idx)
    r_pos=np.zeros([length],dtype=np.float32)
    seq_emb=torch.LongTensor(seq)     # [1,maxlen]
    seqx=embed(seq_emb)    # [1,maxlen,embed_dim]

    for i in range(0,length):
        nxt2=torch.unsqueeze(nxt1[0][i],0)   # [1,embed_dim]
        nxt3=torch.unsqueeze(nxt2,0)   # [1,1,embed_dim]

        actions,seq_r,len_r,seq_d,len_d = Sample_pre(actor,critic,seq_0,
                                          seqx,nxt3,embed)
        # check the lengths of two subsequences
        seq_r1 = torch.LongTensor([seq_r])    # subsequence retained  [1,len_r]
        seq_d1 = torch.LongTensor([seq_d])    # subsequence deleted  [1,len_d]

        seq_rt = embed(seq_r1)  # [1,len_r,embed_dim]
        seq_dt = embed(seq_d1)  # [1,len_d,embed_dim]

        output_r, h1_1, output_d, h2_1 = critic.forward_lstm2(h_r0, c_r0, seq_rt, h_d0, c_d0, seq_dt,"target")

        output1 = h1_1[0]  # the output of positive subsequence through the LSTM

        mlp = MLP(embed_dim,hidden_size)
        nxt = mlp.forward(nxt3[0])

        reward1 = torch.cosine_similarity(output1[0], nxt[0], dim=0)
        r1=reward1.item()
        r_pos[i] = r1
        rewardlist[nxt_0[i]].append(r1)
    idx_sort=r_pos.argsort()

    return idx_sort

    # rank_list = np.zeros([topk], dtype=np.int32)
    # rank=0
    # for i in range(0,length):
    #     if item_idx[idx_sort[i]]==item_idx[0]:
    #        rank=i
    # if rank < topk:
    #     ndcg+=1/np.log2(rank+2)
    #     hit+=1
    # return ndcg,hit

def evaluate_valid(actor, critic, dataset, embed):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    ndcg = 0.0
    ht = 0.0
    valid_user = 0.0

    if usernum>10000:
        users=random.sample(range(1,usernum+1),10000)
    else:
        users=range(1,usernum+1)

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]

        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        valid_user+=1

        idx_sort = predict(actor, critic, u, seq, item_idx, embed)
        length = len(item_idx)
        rank_list = np.zeros([length], dtype=np.int32)

        for i in range(0, length):
            rank_list[i] = item_idx[idx_sort[i]]
            if item_idx[idx_sort[i]] == item_idx[0]:
                rank = i
        if rank < topk:
            ndcg += 1 / np.log2(rank + 2)
            ht += 1
        #print("u:", u)
        #print("seq:", seq)
        #print("item_idx:", item_idx[:10])
        #print("rank_list:", rank_list[:10])

    return ndcg/valid_user,ht/valid_user


def evaluate(actor,critic,dataset,embed):
    [train,valid,test,usernum,itemnum]=copy.deepcopy(dataset)
    ndcg=0.0
    ht=0.0
    test_user=0.0

    if usernum>10000:
        users=random.sample(range(1,usernum+1),10000)
    else:
        users=range(1,usernum+1)

    for u in users:
        if len(train[u])<1 or len(test[u])<1:continue

        seq=np.zeros([maxlen],dtype=np.int32)
        idx=maxlen-1
        seq[idx]=test[u][0]
        idx-=1
        for i in reversed(train[u]):
            seq[idx]=i
            idx-=1
            if idx==-1:
                break
        rated=set(train[u])
        rated.add(0)
        item_idx=[test[u][0]]    # candidate item
        for _ in range(100):
            t=np.random.randint(1,itemnum+1)
            while t in rated:
                t=np.random.randint(1,itemnum+1)
            item_idx.append(t)   # negative sampling

        test_user+=1
        idx_sort = predict(actor, critic, u, seq, item_idx, embed)
        length = len(item_idx)
        rank_list = np.zeros([length], dtype=np.int32)

        for i in range(0, length):
            rank_list[i]=item_idx[idx_sort[i]]
            if item_idx[idx_sort[i]] == item_idx[0]:
                rank = i
        if rank < topk:
            ndcg += 1 / np.log2(rank + 2)
            ht += 1
        #print("u:",u)
        #print("seq:",seq)
        #print("item_idx:",item_idx[:10])
        #print("rank_list:",rank_list[:10])

    return ndcg/test_user,ht/test_user


def main():
    random.seed(seed)

    dataset = data_partition(data_file)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print("usernum:",usernum)
    print("itemnum:",itemnum)
    num_batch = int(math.floor(len(user_train) / batch_size))
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f=open(os.path.join('log.txt'),'w')

    embed = nn.Embedding(itemnum+1,embed_dim)
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

        if epoch >= 0:
            t1=time.time()-t0
            T+=t1
            print("Evaluating...")
            ndcg_valid,hit_valid = evaluate_valid(actor,critic,dataset,embed)
            # ndcg_test,hit_test = evaluate(actor,critic,dataset,embed)
            print('')
            print('epoch:%d, time:%f(s),valid(NDCG@10: %.4f, HR@10: %.4f)' % (
            epoch, T, ndcg_valid, hit_valid))
            f.write('epoch:'+str(epoch+1)+'valid(NDCG@10:'+str(ndcg_valid)+',hit@10:'+str(hit_valid))
            f.flush()
            # print('epoch:%d, time:%f(s),valid(NDCG@10: %.4f, HR@10: %.4f), test(NDCG@10: %.4f, HR@10:%.4f)' % (epoch,T,ndcg_valid,hit_valid,ndcg_test,hit_test))


    f.close()

if __name__=='__main__':
    main()
