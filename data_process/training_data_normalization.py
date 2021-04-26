


import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import gzip
import pickle

from  datetime import *
import time
theano.config.floatX= 'float32'

# start-snippet-1
class dA_joint(object):
    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            input1=None,
            input2=None,
            input3=None,
            n_visible1=4096,
            n_visible2=4096,
            n_hidden=1024,
            W1=None,
            bhid1=None,
            bvis1=None,
            W2=None,
            bhid2=None,
            bvis2=None,
            lamda=None,
            mu=None,
            momentum=0.9
    ):
        self.n_visible1 = n_visible1
        self.n_visible2 = n_visible2
        self.n_hidden = n_hidden
        self.lamda = lamda
        self.mu = mu
        self.momentum = momentum

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W1:
            initial_W1 = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible1)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible1)),
                    size=(n_visible1, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W1 = theano.shared(value=initial_W1, name='W1', borrow=True)
        if not W2:
            initial_W2 = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible2)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible2)),
                    size=(n_visible2, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W2 = theano.shared(value=initial_W2, name='W2', borrow=True)

        if not bvis1:
            bvis1 = theano.shared(
                value=numpy.zeros(
                    n_visible1,
                    dtype=theano.config.floatX
                ),
                name='b1p',
                borrow=True
            )
        if not bvis2:
            bvis2 = theano.shared(
                value=numpy.zeros(
                    n_visible2,
                    dtype=theano.config.floatX
                ),
                name='b2p',
                borrow=True
            )

        if not bhid1:
            bhid1 = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b1',
                borrow=True
            )
        if not bhid2:
            bhid2 = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b2',
                borrow=True
            )

        self.W1 = W1
        self.W2 = W2

        self.b1 = bhid1
        self.b2 = bhid2

        self.b1_prime = bvis1
        self.b2_prime = bvis2

        self.W1_prime = self.W1.T
        self.W2_prime = self.W2.T

        self.theano_rng = theano_rng
        self.L2_sqr = (
            (self.W1 ** 2).mean() + (self.W2 ** 2).mean() + (self.b1 ** 2).mean() + (self.b2 ** 2).mean() +
            (self.W1_prime ** 2).mean() + (self.W2_prime ** 2).mean() + (self.b1_prime ** 2).mean() + (
            self.b2_prime ** 2).mean()
        )
        # if no input is given, generate a variable representing the input
        if input1 is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x1 = T.dmatrix(name='input1',dtype='float32')
            self.x2 = T.dmatrix(name='input2',dtype='float32')
            self.x3 = T.dmatrix(name='input3',dtype='float32')

        else:
            self.x1 = input1
            self.x2 = input2
            self.x3 = input3

        self.params = [self.W1, self.b1, self.b1_prime,self.W2
                       , self.b2, self.b2_prime
                       ]

        # end-snippet-1
        self.output1 = T.nnet.relu(T.dot(self.x1, self.W1) + self.b1, 0.01)
        self.output2 = T.nnet.relu(T.dot(self.x2, self.W2) + self.b2, 0.01)
        self.output3 = T.nnet.relu(T.dot(self.x3, self.W2) + self.b2, 0.01)

        self.output1t = T.transpose(self.output1)
        self.output2t = T.transpose(self.output2)
        self.output3t = T.transpose(self.output3)

        self.rec1 = T.nnet.relu(T.dot(self.output1, self.W1_prime) + self.b1_prime, 0.01)
        self.rec2 = T.nnet.relu(T.dot(self.output2, self.W2_prime) + self.b2_prime, 0.01)
        self.rec3 = T.nnet.relu(T.dot(self.output3, self.W2_prime) + self.b2_prime, 0.01)

    def get_corrupted_input(self, input1, input2, input3, corruption_level):

        a = self.theano_rng.binomial(size=input1.shape, n=1,
                                     p=1 - corruption_level,
                                     dtype=theano.config.floatX) * input1
        b = self.theano_rng.binomial(size=input2.shape, n=1,
                                     p=1 - corruption_level,
                                     dtype=theano.config.floatX) * input2
        c = self.theano_rng.binomial(size=input3.shape, n=1,
                                     p=1 - corruption_level,
                                     dtype=theano.config.floatX) * input3
        return a, b, c

    def get_hidden_values(self, input1, input2, input3):
        """ Computes the values of the hidden layer """
        return T.nnet.relu(T.dot(input1, self.W1) + self.b1, 0.01), T.nnet.relu(
            T.dot(input2, self.W2) + self.b2, 0.01), T.nnet.relu(
            T.dot(input3, self.W2) + self.b2, 0.01)

    def get_reconstructed_input(self, hidden1, hidden2, hidden3):

        a = T.nnet.relu(T.dot(hidden1, self.W1_prime) + self.b1_prime, 0.01)
        b = T.nnet.relu(T.dot(hidden2, self.W2_prime) + self.b2_prime, 0.01)
        c = T.nnet.relu(T.dot(hidden3, self.W2_prime) + self.b2_prime, 0.01)
        return a, b, c

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        y1, y2, y3 = self.get_hidden_values(self.x1, self.x2, self.x3)
        y2t = T.transpose(y2)
        y3t = T.transpose(y3)
        z1, z2, z3 = self.get_reconstructed_input(y1, y2, y3)
        L_x1 = T.mean((z1 - self.x1) ** 2)
        L_x2 = T.mean((z2 - self.x2) ** 2)
        L_x3 = T.mean((z3 - self.x3) ** 2)

        d_x1_x2 = T.dot(y1, y2t).diagonal()
        d_x1_x3 = T.dot(y1, y3t).diagonal()

        L_sup = T.mean(T.nnet.sigmoid(d_x1_x2 - d_x1_x3))

        L_sqr = self.L2_sqr
        L_123 = (L_x1 + L_x2 + L_x3)
        L_rec = self.mu * L_123 + self.lamda * L_sqr
        cost = L_rec - L_sup

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        # updates = [
        #     (param, param - learning_rate * gparam)
        #     for param, gparam in zip(self.params, gparams)
        #     ]
        updates = []
        for p, g in zip(self.params, gparams):
            mparam_i = theano.shared(numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))
            v = self.momentum * mparam_i - learning_rate * g
            updates.append((mparam_i, v))
            updates.append((p, p + v))

        return (cost, updates, L_rec, L_sup, L_sqr, L_123, d_x1_x2, d_x1_x3)


def test_dA(learning_rate=0.1, batch_size=64, epoch_time=30, max_patience=3):
    # x = numpy.array([[1,  2,   3],
    #               [ 3,   5,  6],
    #               [ 7,  8, 9]], dtype='float32')
    # print  x.max(axis=0)
    # print  x.min(axis=0)
    #
    #
    # print x_normed

    print 'loading test visual  data'
    print('now():' + str(datetime.now()))
    with open("AUC_new_dataset_train_811.pkl", "rb") as f:
        print 'test'
        train_set = numpy.asarray(pickle.load(f),dtype='float32')

    with open("AUC_new_dataset_valid_811.pkl", "rb") as f:
        print 'test'
        valid_set = numpy.asarray(pickle.load(f),dtype='float32')

    with open("AUC_new_dataset_test_811.pkl", "rb") as f:
        print 'test'
        test_set = numpy.asarray(pickle.load(f),dtype='float32')

    print 'done'
    row1=train_set[0].shape[0];
    row2=valid_set[0].shape[0];
    row3=test_set[0].shape[0];

    print 'row1'
    print row1
    print 'row2'
    print row2
    print 'row3'
    print row3


    def Normalization(x):
            A= numpy.array(x.max(axis=0)- x.min(axis=0))
            A[numpy.where(A<0.000000000000000001)]=0.000000001;
            return  (x- x.min(axis=0) )/A


    all_data_xi=numpy.vstack((train_set[0],valid_set[0]))
    all_data_xi=numpy.vstack((all_data_xi, test_set[0]))
    print 'all data xi'
    print all_data_xi.shape
    print 'processing xi'
    print('now():' + str(datetime.now()))
    new_data_xi= Normalization(all_data_xi)

    all_data_xj=numpy.vstack((train_set[1],valid_set[1]))
    all_data_xj=numpy.vstack((all_data_xj, test_set[1]))
    print 'all data xj'
    print all_data_xj.shape
    print 'processing xj'
    print('now():' + str(datetime.now()))
    new_data_xj= Normalization(all_data_xj)

    all_data_xk=numpy.vstack((train_set[2],valid_set[2]))
    all_data_xk=numpy.vstack((all_data_xk, test_set[2]))
    print 'all data xk'
    print all_data_xk.shape
    print 'processing xk'
    print('now():' + str(datetime.now()))
    new_data_xk= Normalization(all_data_xk)


    write_file=open('AUC_new_dataset_train_811_norm.pkl','wb')
    pickle.dump([new_data_xi[0:row1,:],new_data_xj[0:row1,:],new_data_xk[0:row1,:]],write_file)
    write_file.close()
    print new_data_xi[0:row1,:].shape


    write_file=open('AUC_new_dataset_valid_811_norm.pkl','wb')
    pickle.dump([new_data_xi[row1:row1+row2,:],new_data_xj[row1:row1+row2,:],new_data_xk[row1:row1+row2,:]],write_file)
    write_file.close()
    print new_data_xi[row1:row1+row2,:].shape

    write_file=open('AUC_new_dataset_test_811_norm.pkl','wb')
    pickle.dump([new_data_xi[row1+row2:row1+row2+row3,:],new_data_xj[row1+row2:row1+row2+row3,:],new_data_xk[row1+row2:row1+row2+row3,:]],write_file)
    write_file.close()
    print new_data_xi[row1+row2:row1+row2+row3,:].shape

if __name__ == '__main__':
    test_dA()
