import scipy.io as sio
import random
import numpy as np
DEGUG = False
matfn = 'cifar_10_gist.mat'
data=sio.loadmat(matfn)



if DEGUG:
    data_name = []
    for i in data:
        if i[0]=='_':
            continue
        data_name.append(i)
    # print data_name
    # print data['traingnd'][:20]
    # print data['traindata'].shape
    print data_name
def get_W():
    return sio.loadmat('W.mat')['pc']

def get_train_data():

    return  data['traindata']


def get_test_data():
    return  data['testdata']

def get_train_gt():
    return  data['traingnd']

def get_test_gt():
    return data['testgnd']

if '__main__'==__name__:
    print get_test_data().shape
    print get_train_data().shape
    print get_train_gt().shape
    print get_test_gt().shape
    print get_W().shape

