import sys
import copy
import random
from collections import defaultdict
import numpy as np
from multiprocessing import Process,Queue
import pickle


def data_partition(fname1):
    usernum=0
    itemnum=0
    userlist=[]   #计算user的个数
    itemlist=[]   #计算item的个数
    User=defaultdict(list)
    Usertest=defaultdict(list)

    user_train={}
    user_valid={}
    user_test={}
    item_strtoint_dict={}
    user_strtoint_dict={}
    f=open('data/%s.txt'%fname1,'r')
    for line in f:
        u,i=line.rstrip().split(' ')
        if u not in userlist:
            userlist.append(u)
        if i not in itemlist:
            itemlist.append(i)
        usernum=len(userlist)
        itemnum=len(itemlist)

    f.close()

    for i in range(len(userlist)):
        user_strtoint_dict[userlist[i]]=i
    pickle.dump(user_strtoint_dict,open('user_strtoint_dict','wb'))

    for i in range(len(itemlist)):
        item_strtoint_dict[itemlist[i]] = i+1
    pickle.dump(item_strtoint_dict, open('item_strtoint_dict', 'wb'))

    f1=open('data/%s.txt'%fname1,'r')
    for line in f1:
        u_1,i_1=line.rstrip().split(' ')
        User[user_strtoint_dict[u_1]].append(item_strtoint_dict[i_1])   #生成的sequence是id号，用于取embedding
    f1.close()

    for user in User:
        user_train[user]=User[user][:-2]
        user_valid[user]=User[user][-2]
        user_test[user]=User[user][-1]

    return [user_train,user_valid,user_test,usernum,itemnum+1]
