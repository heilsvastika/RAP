import os
import time
import argparse
import tensorflow as tf
import math
from tensorflow.python.framework import ops
from sampler import WarpSampler
from util import *
from sampler import *
from model import *
from critic import CriticNetwork
from actor import ActorNetwork

def sampling_RL(sess,args,actor,critic,inputs,vec,pos,length,epsilon=0.,Random=True):
    pad=np.zeros((1,args.dim),dtype=np.float32)
    padding=tf.convert_to_tensor(pad)
    nxt=pos[-1]
    nxt_vec=critic.create_vec(nxt)
    current_lower_state=tf.concat([padding,nxt_vec],axis=1)
    vec=critic.create_vec(inputs)
    actions=[]
    states=[]
    #sampling actions
    for i in range(length):
        predicted=actor.predict_target(current_lower_state,vec[0][i])
        states.append([current_lower_state,vec[0][pos]])
        if Random:
            if random.random() > epsilon:
                action=(0 if random.random()<predicted[0] else 1)
            else:
                action=(1 if random.random()<predicted[0] else 0)
        else:
            action=np.argmax(predicted)
        actions.append(action)
        if action==1:
            out_d,current_lower_state=critic.lower_LSTM_target(current_lower_state,[[inputs[i]]])
    Rinput=[]
    Rinput_del=[]
    for (i,a) in enumerate(actions):
        if a==1:
            Rinput.append(inputs[i])
        else:
            Rinput_del.append(inputs[i])
    Rlength=len(Rinput)
    Rlength_del=len(Rinput_del)
    if Rlength==0:
        actions[-1]=1
        Rinput.append(inputs[-1])
        Rlength=1
    Rinput+=[0]*(args.maxlen-Rlength)
    return actions,states,Rinput,Rlength,Rinput_del,Rinput_del


def train(sess,args,actor,critic,sampler,dataset,batch_size,samplecnt=5,LSTM_trainable=True,RL_trainable=True):
    #print("training :total ",len(user_train),"nodes.")
    #random.shuffle(user_train)

    [user_train,user_valid,user_test,usernum,itemnum]=copy.deepcopy(dataset)
    sess.run(tf.global_variables_initializer())
    for b in range(int(math.floor(len(user_train)/batch_size))):
        u,seq,pos,_= sampler.next_batch()
        totloss=0.
        critic.assign_active_network()
        actor.assign_active_network()
        for i in range(batch_size):
            seq=seq[i]
            length=len(seq)

            #train the predict network
            if RL_trainable:
                actionlist,statelist,losslist=[],[],[]
                aveloss=0.
                for j in range(samplecnt):
                    actions,states,Rinput,Rlength,Rinput_del,Rlength_del=sampling_RL(sess,args,actor,critic,seq[i],critic.create_vec([seq[i]]),pos[i][-1],length,epsilon=0.,Random=True)
                    actionlist.append(actions)
                    statelist.append(states)
                    out1,loss1=critic.getloss([Rinput],[Rlength],[pos[i][-1]])
                    out2,loss2=critic.getloss([Rinput_del],[Rlength_del],[pos[i][-1]])
                    p1=out1[pos[i][-1]-1]
                    p2=out2[pos[i][-1]-1]
                    aveloss+=loss1+loss2
                    losslist.append(loss1+loss2)
                    reward1=tf.log(p1)
                    reward2=tf.log(p2)
                    reward=reward1-reward2
                    g=actor.get_gradient(statelist[i][pos][0],statelist[i][pos[1]],reward)
                    if grad == None:
                        grad=g
                    else:
                        grad[0]+=g[0]
                        grad[1]+=g[1]
                        grad[2]+=g[2]
                actor.train(grad)
                aveloss/=samplecnt
                totloss+=aveloss

                grad=None
        if RL_trainable:
            actor.update_target_network()
            if LSTM_trainable:
                critic.update_target_network()
        else:
            critic.assign_target_network()




def test(sess,args,actor,critic,dataset,noRL=False):
    [train,valid,test,usernum,itemnum]=copy.deepcopy(dataset)
    ndcg=hit=0.0
    test_user=0.0
    list_test_user=list(range(0,usernum))
    if usernum>1000000:
        users=random.sample(list_test_user,20000)
    else:
        users=list_test_user
    for u in users:
        seq=np.zeros([args.maxlen],dtype=np.int32)
        idx=args.maxlen-1
        seq[idx]=test[u]
        idx-=1
        for i in reversed(train[u]):
            seq[idx]=i
            idx-=1
            if idx == -1:
                break
            rated=set(train[u])
            item_idx=[test[u]]
            for _ in range(999):
                t=np.random.randint(0,itemnum)
                while t in rated:
                    t=np.random.randint(0,itemnum)
                item_idx.append(t)
            actions,states,Rinput,Rlength,Rinput_del,Rlength_del=sampling_RL(sess,args,actor,critic,[seq],critic.create_vec(seq),[item_idx[0]])
            predictions=-critic.predict_target([Rinput],[Rlength],[item_idx[0]])
            #out2=critic.predict_target([Rinput_del],[Rlength_del],[item_idx[0]])
            item_idx=np.array(item_idx)
            pred=item_idx[np.argsort(predictions[0])]
            test_user+=1
            if item_idx[0] in pred[:10]:
                hit+=1
            idcg=np.sum(1/np.log(np.arange(2,1+2)))
            dcg=0.0
            for i,p in enumerate(pred[:10]):
                if p==item_idx[0]:
                    dcg+=1/np.log2(i+2)
            ndcg+=dcg/idcg
            if test_user % 100 == 0:
                sys.stdout.flush()
    return hit/test_user,ndcg/test_user


def main():
    def str2bool(s):
        if s not in {'False','True'}:
            raise ValueError('Not a valid boolean string')
        return s=='True'

    parser=argparse.ArgumentParser()
    parser.add_argument('--datatrain',default='ml-1m',type=str)
    parser.add_argument('--train_dir',default='-1',type=str)
    parser.add_argument('--batch_size',default=128,type=int)
    parser.add_argument('--lr',default=0.0001,type=float)
    parser.add_argument('--maxlen',default=100,type=int)
    parser.add_argument('--dim',default=100,type=int)
    parser.add_argument('--dropout',default=0.005,type=float)
    parser.add_argument('--l2_emb',default=0.0,type=float)
    parser.add_argument('--optimizer',default='Adam',type=str)
    args=parser.parse_args()

    if not os.path.isdir(args.datatrain+'_' +args.train_dir):
        os.makedirs(args.datatrain+'_'+args.train_dir)

    dataset=data_partition(args.datatrain)
    [user_train,user_valid,user_test,usernum,itemnum]=dataset

    num_batch=int(math.floor(len(user_train)/args.batch_size))
    cc=0.0
    for u in user_train:
        cc+=len(user_train[u])
    print('average sequence length:%.2f'%(cc/len(user_train)))

    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    with tf.Graph().as_default(),tf.Session() as sess:
        model=Model(usernum, itemnum, args, reuse=None)
        sess.run(tf.global_variables_initializer())
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        totloss=0.0
        sampler=WarpSampler(user_train,usernum,itemnum,batch_size=args.batch_size,maxlen=args.maxlen,n_workers=3)
        for epoch in range(1,args.num_epochs+1)
            u, seq, pos, _ = sampler.next_batch()
            u = np.array(u)
            seq = np.array(seq)
            seq,table=model.embed_seq(sess,seq)

            #model
            critic = CriticNetwork(sess, args, table)
            print("num_other_variables:",critic.num_other_variables)
            print("len(network params):",len(critic.network_params))
            print("len(target network params):",len(critic.target_network_params))
            actor = ActorNetwork(sess,args)

            saver=tf.train.Saver()
            #for item in tf.trainable_variables():
            #    print(item.name,item.get_shape())
            sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

            train(sess,args,actor,critic,sampler,dataset,samplecnt=5,LSTM_trainable=True,RL_trainable=True)
            print("epoch ", epoch, "total loss:", totloss)
            critic.assign_target_network()

            hit1,ndcg1=test(sess,actor,critic,user_valid,True)
            hit2,ndcg2=test(sess,actor,critic,user_test,True)

            if b % 5 == 0:
                hit1, ndcg1 = test(sess, actor, critic, user_test)
                hit2, ndcg2 = test(sess, actor, critic, user_test)
                print("epoch ", b, "total loss:", totloss, "-----valid----ndcg:", ndcg1, ",hit:", hit1,
                  "-----test----ndcg:", ndcg2, ",hit:", hit2)

if __name__=="__main__":
    main()