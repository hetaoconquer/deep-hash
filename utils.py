import numpy as np
import theano.tensor as T

def pca(data,k):
    from numpy import mean,cov,linalg,transpose
    m = mean(data,axis=0)
    data -= m
    C = cov(transpose(data))
    evals,evecs = linalg.eig(C)
    list_val = np.argsort(evals)
    topk = list_val[-1:-(k+1):-1]
    return evecs[:, topk]


def ortho_weight(ndim):
    W = rng_numpy.randn(ndim, ndim)
    u, _, _ = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.01, ortho=False):
    rng_numpy = np.random.RandomState(1234)
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * rng_numpy.randn(nin, nout)
    return W.astype('float64')

def abs(a):
    if a < 0:
        return -a
    else:
        return a


# def get_accucy(train_data_code,train_label,pre_code,pre_label,N=500):
#     ret = 0.
#     for pj, pl in zip(pre_code,pre_label):
#         ans = []
#         lab = []
#         for i, l in zip(train_data_code,train_label):
#            dist = hamingDist(pj,i)
#            lab.append(l)
#            ans.append(dist)
#         ind = np.argsort(ans)
#         ind = ind[:N]
#         lab = np.array(lab)
#         lab_ = lab[ind]
#         ret += ((np.where(lab_==pl)[0]).shape[0])*1./(N*1.)
#     return ret/1000.

def cat_map(train_data_code,train_label,pre_code,pre_label,N=500):
    size = pre_code.shape[0]
    sim = np.dot(train_data_code, np.transpose(pre_code))
    inx = np.argsort(-sim, 0)
    inx = inx[:N]
    lab = np.array(train_label)
    pre_label = np.array(pre_label)
    nn_lab = lab[inx]
    apall = 0
    for i in range(size):
        x=0
        p=0
        new_label = nn_lab[:,i]
        for j in range(N):
            if new_label[j]==pre_label[i]:
                x = x + 1
                p = p + x*1./(j+1)
        if x:
            apall += p/N
    return apall/size

def get_pre(train_data_code,train_label,pre_code,pre_label,N=1000):
    size = pre_code.shape[0]
    sim = np.dot(train_data_code,np.transpose(pre_code))
    inx = np.argsort(-sim,0)
    inx = inx[:N]
    lab = np.array(train_label)
    pre_label = np.array(pre_label)
    nn_lab = lab[inx]
    pre = 0.
    for i in range(size):
        pre = pre + np.sum(nn_lab[:,i]==pre_label[i])*1./N
    return pre/size

def get_2_pre(train_data_code,train_label,pre_code,pre_label):
    hamDis = pre_code.shape[1]-4
    size = pre_code.shape[0]
    trainsize = train_data_code.shape[0]
    sim = np.dot(train_data_code, np.transpose(pre_code))
    sim  = np.array(sim)
    pre = 0
    for i in range(size):
        n = 0
        t = 0
        for j in range(trainsize):
            if sim[j][i]>=hamDis:
                n+=1
                if train_label[j]==pre_label[i]:
                    t+=1
        if n==0:
            pre += 0
        else:
            pre += t*1./n
    return pre*1./size

def get_2_accucy(train_data_code,train_label,pre_code,pre_label):
    size = pre_code.shape[0]
    ret = 0.
    for pj, pl in zip(pre_code,pre_label):
        ans = 0.
        fenmu = 0.
        for i, l in zip(train_data_code,train_label):
           dist = hamingDist(pj,i)
           if dist<=2 :
                fenmu += 1.
                if l == pl:
                    ans += 1.
        if(fenmu==0):
            size = size-1
        else:
            ret +=  ans / fenmu
    return ret/size



def get_MAP(train_data_code,train_label,pre_code,pre_label,N=500):
    ret = 0.

    for pj, pl in zip(pre_code,pre_label):
        ans = []
        lab = []
        for i, l in zip(train_data_code,train_label):
           dist = hamingDist(pj,i)
           lab.append(l)
           ans.append(dist)
        ind = np.argsort(ans)
        ind = ind[:N]
        lab = np.array(lab)
        lab_ = lab[ind]
        nn = 0
        o = 0.
        for p in xrange(N):
            if lab_[p] == pl:
                nn += 1.
                o += nn/(p+1.)
        ret += (o/nn)
    return ret/pre_label.shape[0]

def hamingDist(a, b):
    dist = 0
    assert len(a) == len(b), 'Two vectors have not the same length ! please check !'
    for i, j in zip(a, b):
        if i != j:
            dist += 1
    return dist

def sqdist(a,b,data_num = 59000,dimen = 80):
    a = T.transpose(a)
    b = T.transpose(b)
    aa = T.reshape(T.sum(a ** 2, 0),(1,data_num))
    bb = T.reshape(T.sum(b ** 2, 0),(1,dimen))
    ab= T.dot(T.transpose(a),b)
    d = T.repeat(T.transpose(aa),bb.shape[1],axis=1) + T.repeat(bb,aa.shape[1],axis = 0) - 2*ab
    sigma = T.mean(d)
    d = T.exp(-d / (2 * sigma))
    mvec = T.reshape(T.mean(d, 0),(1,dimen))
    d = d - T.repeat(mvec, d.shape[0], axis=0)
    return d,sigma,mvec

def sqdistnum(a,b,sigma,mvec):
    a = np.transpose(a)
    b = np.transpose(b)
    aa = np.asmatrix(np.sum(a ** 2, 0))
    bb = np.asmatrix(np.sum(b ** 2, 0))
    ab= np.dot(np.transpose(a),b)
    d = np.repeat(np.transpose(aa),bb.shape[1],axis=1) + np.repeat(bb,aa.shape[1],axis = 0) - 2*ab
    d = np.exp(-d/(2*sigma))
    d -= np.repeat(mvec,d.shape[0],axis = 0)
    return d

def oneHotVector(gt):
    a = []
    for i in gt:
        b = [0]*10
        b[int(i)] = 1
        a.append(b)
    return  np.array(a)

def labelmatrix(trn,batchsize):
    temp = np.repeat(trn, batchsize, axis=1) - np.repeat(np.transpose(trn), batchsize, axis=0)
    s = -np.ones((batchsize, batchsize),dtype='int32')
    s[np.where(temp == 0)] = 1
    return s
