# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

# define a lstm network
class LSTMClassifier(nn.Module):
    def __init__(self, args, itemnum, embed):
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

        self.batch_size = args.batch_size
        self.output_size = itemnum
        self.hidden_size = args.hidden_size
        self.itemnum = itemnum
        self.embed_dim = args.embed_dim
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size,1)     # (input_size,hidden_size,num_layers)
        self.label = nn.Linear(self.hidden_size,self.output_size)
        self.embed=embed

    # obtain the embedding of item
    def ivector_find(self, x):
        return self.embed(x)


    def getNextHiddenState(self,hc,x):
        hidden = hc[0, 0:self.hidden_size].view(1, 1, self.hidden_size)
        cell = hc[0, self.hidden_size:].view(1, 1, self.hidden_size)
        input = self.embed(x)
        input = input.view(1, 1, -1)
        out, hidden = self.lstm(input, [hidden, cell])    # (inputs,[h,c])
        hidden = torch.cat([hidden[0], hidden[1]], -1).view(1, -1)
        return out, hidden
