# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F

# define a lstm network
class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, hidden_size, itemnum, embed_dim,embed,tau):
        super(LSTMClassifier, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        """

        self.batch_size = batch_size
        self.output_size = itemnum
        self.hidden_size = hidden_size
        self.itemnum = itemnum
        self.embed_dim = embed_dim
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size,1,batch_first=True)     # (input_size,hidden_size,num_layers)
        self.label = nn.Linear(self.hidden_size,self.output_size)
        self.embed=embed

    #def forward(self,hc,x):
    #    input = np.array(x)
    #    input = torch.LongTensor([input])
    #    f=torch.zeros([1, 1, self.embed_dim])   #[1,1,embed_dim]
    #    input = self.embed(input)
    #    input = input.view(1, 1, -1)   #[1,1,embed_dim]
    #    print("input.size:",input.shape)
    #    print("hc.size:",hc.shape)
    #    if hc.equal(f):
    #        hidden = torch.zeros([1, 1, self.hidden_size])
    #        cell = torch.zeros([1, 1, self.hidden_size])
    #        out, hidden1 = self.lstm(input, [hidden, cell])  # (input,[h,c])
    #        hidden2 = torch.cat([hidden1[0], hidden1[1]], -1).view(1, -1)
    #    else:
    #        out, hidden2 = self.lstm(input,hc)  # (input,[h,c])
    #    return out,hidden2

    def forward(self,hc,x):
        #hidden = hc[0, 0:self.hidden_size].view(1, 1, self.hidden_size)
        #cell = hc[0, self.hidden_size:].view(1, 1, self.hidden_size)
        #print("hc.size:",hc.shape)
        h=hc[0]
        #print("h.size:",h.shape)
        hidden=h[0,0:self.hidden_size].unsqueeze(0)
        cell = h[0,self.hidden_size:self.embed_dim].unsqueeze(0)

        hidden1 = torch.unsqueeze(hidden, 0)  # [1,1,hidden_size]
        cell1 = torch.unsqueeze(cell, 0)   # [1,1,hidden_size]

        seq = torch.LongTensor([x])  # [1]
        input0 = torch.unsqueeze(self.embed(seq),0)  # [1,1,embed_dim]
        input = input0.view(1,1,-1)   # [1,1,embed_dim]

        #print("cell1.size:", cell1.shape)
        out, hidden2 = self.lstm(input,(hidden1,cell1))    # (inputs,[h,c])
        # hidden = torch.cat([hidden[0], hidden[1]], -1).view(1, -1)
        return out, hidden2


    # obtain the embedding of item
    def ivector_find(self, x):
        return self.embed(x)
