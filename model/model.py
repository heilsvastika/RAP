from util import *
import random
import tensorflow as tf

class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))

        with tf.variable_scope("model", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, self.item_emb_table = embedding(self.input_seq,
                                                     vocab_size=itemnum + 1,
                                                     dim=args.dim,
                                                     zero_pad=True,
                                                     scale=True,
                                                     l2_reg=args.l2_emb,
                                                     scope="input_embeddings",
                                                     with_t=True,
                                                     reuse=None)
            print('seqembedding Done')

            #self.actions,self.states,self.Rinput,self.Rlength,self.Rinput_delete, self.Rlength_delete=sample_RL(actor,critic,self.seq,itemnum+1,args)
            print("actions Done!")

            self.seq = normalize(self.seq)

            self.test_item = tf.placeholder(tf.int32, shape=(1000,))
            test_item_emb = tf.nn.embedding_lookup(self.item_emb_table, self.test_item)

    def embed_seq(self, sess,input_seq):
        return sess.run([self.seq,self.item_emb_table],{self.input_seq:input_seq})