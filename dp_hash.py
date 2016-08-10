import theano
import numpy as np
import theano.tensor as T
import dataio
import scipy.io as sio
from datetime import datetime
import utils

node1 = 100
node2 = 80
node3 = 30
# set the eyes matrix
I1 = np.eye(node1) * 1.
I2 = np.eye(node2) * 1.
I3 = np.eye(node3) * 1.

# set the input and output
x = T.matrix('x')
feature_dim = dataio.get_train_data().shape[1]
trnlabel = dataio.get_train_gt()

tmp1 = sio.loadmat('wat.mat')['a']
w1 = theano.shared(tmp1, name="w1")
#w1 = theano.shared(np.random.uniform(-np.sqrt(1/512.),np.sqrt(1/512.),(512,100 )), name="w1")


train_data = dataio.get_train_data()
m1 = np.mean(train_data,axis=0)
train_data -= m1


w2 = theano.shared(np.random.uniform(-np.sqrt(1/100.),np.sqrt(1/100.),(100,80 )), name="w2")
w3 = theano.shared(np.random.uniform(-np.sqrt(1/80.),np.sqrt(1/80.),(80,30 )), name="w3")


# do the forward propagation
h1 = T.tanh(T.dot(x, w1))
h2 = T.tanh(T.dot(h1, w2))
h3 = T.tanh(T.dot(h2, w3))
B = T.switch(T.lt(h3, 0.), -1., 1.)

lamd1 = 100.
lamd2 = 0.001
lamd3 = 0.001
floss = (0.5 * T.sum((B - h3) ** 2)) - \
       (0.5 * lamd1 * 1. / (train_data.shape[0]) * T.sum((h3 ** 2)))+\
       0.5 * lamd3 * (T.sum(w1 ** 2) + T.sum(w2 ** 2) + T.sum(w3 ** 2))+\
    0.5 * lamd2 * T.sum((T.dot(T.transpose(w1),w1) - I1) ** 2) + \
    0.5 * lamd2 * T.sum((T.dot(T.transpose(w2),w2 ) - I2) ** 2) + \
    0.5 * lamd2 * T.sum((T.dot(T.transpose(w3),w3) - I3) ** 2)

dw1, dw2, dw3 = T.grad(floss, [w1, w2, w3])
rate = 0.01
train = theano.function(
        inputs=[x],
        outputs=[floss,B],
        updates=(
                 (w1, w1 - rate * dw1),
                 (w2, w2 - rate * dw2),
                 (w3, w3 - rate * dw3)

                 )
                       )

predict = theano.function(
        inputs=[x],
        outputs=[B,floss]
)

it = 0
last_J = 0
epsilon = 10e-4 * 0.5
DEBUG = False
file = open('dp_hash.log', 'w+')
# slabel = utils.labelmatrix(trnlabel[0:batch_size], batch_size)
while it<300:
    J,Bx = train(train_data)
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    it += 1
    file.writelines("%s  iterators : %d, loss : %f eption : %f \n" % (time, it, J, abs(J - last_J)))
    print("%s  iterators : %d, loss : %f eption : %f " % (time, it, J, abs(J - last_J)))
    last_J = J


h1 = np.tanh(np.dot(train_data, w1.get_value()))
h2 = np.tanh(np.dot(h1, w2.get_value()))
h3 = np.tanh(np.dot(h2, w3.get_value()))
Bx = np.sign(h3)

test = dataio.get_test_data()
mm = np.mean(test,axis=0)
test -= mm
h1 = np.tanh(np.dot(test, w1.get_value()))
h2 = np.tanh(np.dot(h1, w2.get_value()))
h3 = np.tanh(np.dot(h2, w3.get_value()))
tBx = np.sign(h3)

# Bx = predict(dataio.get_train_data())[0]
print Bx.shape
# tBx = predict(dataio.get_test_data())[0]
print tBx.shape
# np.savez('WandC_No_Bias', w1=w1.get_value(), w2=w2.get_value(),w3=w3.get_value(),b1=b1.get_value(),b2=b2.get_value(),b3=b3.get_value())
# map = utils.get_MAP(Bx,dataio.get_train_gt(),predict(dataio.get_test_data())[0],dataio.get_test_gt(),N=1000)
# print "MAP : ",map
sio.savemat('code.mat', {'traincode':Bx,'testcode':tBx})
map = utils.cat_map(Bx,dataio.get_train_gt(),tBx,dataio.get_test_gt(),N=1000)
print "MAP : ",map
pre = utils.get_pre(Bx,dataio.get_train_gt(),tBx,dataio.get_test_gt(),N=1000)
print "PRE : ",pre
# pre2 = utils.get_2_pre(Bx,dataio.get_train_gt(),tBx,dataio.get_test_gt())
# print "PRE2 : ",pre2
# ret = utils.get_2_accucy(Bx,dataio.get_train_gt(),predict(dataio.get_test_data())[0],dataio.get_test_gt())
# print "HAM2 : ",ret
#file.writelines("accuracy : %f \n" %(ret))



