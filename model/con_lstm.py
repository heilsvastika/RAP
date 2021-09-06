import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

# define a con_lstm network
class CON_LSTM(nn.Module):
    def __init__(self,batch_size,hidden_size,itemnum,embed_dim,embed):
        super(CON_LSTM,self).__init__()
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
        self.lstm1 = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size,num_layers=1,batch_first = True)  # (input_size,hidden_size,num_layers)
        self.lstm2 = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size,num_layers=1,batch_first = True)

        batch_first = True
        self.label = nn.Linear(self.hidden_size, self.output_size)
        self.embed = embed

    def ivector_find(self,x):
        return self.embed(x)

    def getNextHiddenState1(self,hc,x):
        hidden = hc[0, 0:self.hidden_size].view(1, 1, self.hidden_size)
        cell = hc[0, self.hidden_size:].view(1, 1, self.hidden_size)

        input = self.embed(x)
        input = input.view(1,1,-1)
        out,hidden = self.lstm1(input,[hidden, cell])

        hidden = torch.cat([hidden[0], hidden[1]], -1).view(1,-1)

        return out,hidden

    def getNextHiddenState(self,h1,c1,x1,h2,c2,x2):
        seq_r1=x1
        seq_r2=x2

        output1, (h1_n, c1_n) = self.lstm1(seq_r1, (h1, c1))
        output2, (h2_n, c2_n) = self.lstm2(seq_r2, (h2, c2))

        return output1,h1_n,output2,h2_n
