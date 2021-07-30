import argparse
import time

class Parser1(object):
    def getParser(self):
        parser1=argparse.ArgumentParser()
        parser1.add_argument('--datatrain', default='ml-1m', type=str)
        parser1.add_argument('--batch_size', default=128, type=int)
        parser1.add_argument('--lr', default=0.001, type=float)
        parser1.add_argument('--maxlen', default=5, type=int)
        parser1.add_argument('--hidden_size', default=50, type=int, help='hidden state size of gru module')
        parser1.add_argument('--embed_dim', type=int, default=100, help='the dimension of item embedding')
        parser1.add_argument('--num_epochs', default=200, type=int)
        parser1.add_argument('--dropout_rate', default=0.005, type=float)
        parser1.add_argument('--l2_emb', default=0.0, type=float)
        parser1.add_argument('--test', action='store_true', help='test')
        parser1.add_argument('--topk', type=int, default=10)
        parser1.add_argument('--tau', default=0.1, type=float)
        parser1.add_argument('--seed',type=int,default=int(1000*time.time()))
        parser1.add_argument('--samplecnt',type=int,default=5)
        parser1.add_argument('--epsilon',type=float,default=0.05)
        parser1.add_argument('--alpha',type=float,default=0.1)
        return parser1
