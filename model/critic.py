import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import tflearn
import numpy as np

class CriticNetwork(object):
    """
    predict network
    use the word vector and actions(sampled from actor network)
    get the final prediction
    """

    def __init__(self,sess,args,table):
        self.global_step=tf.Variable(0,trainable=False,name="LSTMStep")
        self.sess=sess
        self.maxlen=args.maxlen
        self.dim=args.dim
        self.lr=args.lr
        #self.tau=args.tau
        #self.grained=grained
        self.dropout=args.dropout
        self.init=tf.random_uniform_initializer(-0.05,0.05,dtype=tf.float32)
        self.l2regular=0.00001
        self.tau=0.1
        self.grained=3418
        self.optimizer=tf.train.AdamOptimizer(self.lr)
        self.keep_prob=tf.placeholder(tf.float32,name="keepprob")
        self.num_other_variables=len(tf.trainable_variables())
        self.table=tf.get_variable('table',dtype=tf.float32,initializer=table,trainable=True)
        self.target_table = tf.get_variable('table_target', dtype=tf.float32, initializer=table, trainable=True)

        print("CriticNetwork Done!")

        #lstm cells
        self.lower_cell_state,self.lower_cell_input,self.lower_cell_output,self.lower_cell_state1=self.create_LSTM_cell("Lower/Active")

        # critic network(updating)
        self.inputs,self.length,self.out=self.create_critic_network("Active")
        self.network_params=tf.trainable_variables()[self.num_other_variables:]

        #lstm cells
        self.target_lower_cell_state,self.target_lower_cell_input,self.target_lower_cell_output,self.target_lower_cell_state1=self.create_LSTM_cell("Lower/Target")

        # critic network(delayed updating)
        self.target_inputs,self.target_length,self.target_out=self.create_critic_network("Target")
        self.target_network_params=tf.trainable_variables()[len(self.network_params)+self.num_other_variables:]

        #delayed updating critic networks ops
        self.update_target_network_params=[self.target_network_params[i].assign(tf.multiply(self.network_params[i],self.tau)+tf.multiply(self.target_network_params[i],1-self.tau)) for i in range(len(self.target_network_params))]


        self.assign_target_network_params=[self.target_network_params[i].assign(self.network_params[i]) for i in range(len(self.target_network_params))]
        self.assign_active_network_params=[self.network_params[i].assign(self.target_network_params[i]) for i in range(len(self.network_params))]

        self.ground_truth=tf.placeholder(tf.float32,[1,self.grained],name="ground_truth")

        #total variables
        self.num_trainable_variables=len(self.network_params)+len(self.target_network_params)

        self.ground_truth = tf.placeholder(tf.float32, [1, self.grained], name="ground_truth")

        # calculate loss
        self.loss_target = tf.nn.softmax_cross_entropy_with_logits(labels=self.ground_truth, logits=self.target_out)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.ground_truth, logits=self.out)
        self.loss2 = 0
        with tf.variable_scope("Lower/Active", reuse=True):
            self.loss2 += tf.nn.l2_loss(tf.get_variable('lstm_cell/kernel'))
        with tf.variable_scope("Active/pred", reuse=True):
            self.loss2 += tf.nn.l2_loss(tf.get_variable('W'))
            self.loss += self.loss2 * self.L2regular
            self.gradients = tf.gradients(self.loss_target, self.target_network_params)
            self.optimize = self.optimizer.apply_gradients(zip(self.gradients, self.network_params),
                                                       global_step=self.global_step)

    def create_LSTM_cell(self,Scope):
        cell=LSTMCell(self.dim,initializer=self.init,state_is_tuple=False)
        state=tf.placeholder(tf.float32,shape=[1,cell.state_size],name="cell_state")      # state: [1,200]
        inputs=tf.placeholder(tf.int32,shape=[1,1],name="cell_input")     # inputs: [1,1]
        if Scope[-1]=='e':
            vec=tf.nn.embedding_lookup(self.table,inputs)    # table: [3418,100]    inputs: [1,1]
        else:
            vec=tf.nn.embedding_lookup(self.target_table,inputs)      # target_table: [3418,100]    inputs: [1,1]
        with tf.variable_scope(Scope,reuse=False):
            out,state1=cell(vec[:,0,:],state)    # vec:[1,1,100], vec[:,0,:]:[1,100],   state: [1,200]
        return state,inputs,out,state1


    def create_critic_network(self,Scope):
        inputs=tf.placeholder(shape=[1,self.maxlen],dtype=tf.int32,name="inputs")    #inputs是输入的sequence，maxlen:10
        length=tf.placeholder(shape=[1],dtype=tf.int32,name="length")

        # Lower network
        if Scope[-1]=='e':
            vec=tf.nn.embedding_lookup(self.table,inputs)    # vec:[1,10,100],  table:[3418,100],   inputs:[1,10]
        else:
            vec=tf.nn.embedding_lookup(self.target_table,inputs)
        cell=LSTMCell(self.dim,initializer=self.init,state_is_tuple=False)

        with tf.variable_scope("Lower",reuse=True):
            out,_=tf.nn.dynamic_rnn(cell,vec,length,dtype=tf.float32,scope=Scope)     # out:[1,10,100],   out[0]:[10,100]
        out=tf.gather(out[0],length-1)
        out=tflearn.dropout(out,self.keep_prob)
        out=tflearn.fully_connected(out,self.grained,scope=Scope+"/pred",name="get_pred")    # out:[1,grained]

        return inputs,length,out


    def assign_active_network(self):
        self.sess.run(self.assign_active_network_params)

    def get_num_trainable_variables(self):
        return self.get_num_trainable_variables


    def create_vec(self,input):
        vec=tf.nn.embedding_lookup(self.target_table,[input])
        return vec

    def getloss(self, inputs, length, ground_truth):
        return self.sess.run([self.target_out, self.loss_target], feed_dict={
            self.target_inputs: inputs,
            self.target_ground_truth: ground_truth,
            self.keep_prob: 1.0})

    def predict_target(self, inputs, length):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_length: length,
            self.keep_prob: 1.0
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def assign_target_network(self):
        self.sess.run(self.assign_target_network_params)

    def assign_active_network(self):
        self.sess.run(self.assign_active_network_params)

    def get_num_trainable_vars(self):
        return self.get_num_trainable_vars

    def lower_LSTM_target(self, state, inputs):
        return self.sess.run([self.target_lower_cell_output,
                              self.target_lower_cell_state1], feed_dict={
                             self.target_lower_cell_state:state,
                             self.target_lower_cell_input:inputs})