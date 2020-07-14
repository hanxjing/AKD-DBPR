# coding:utf-8
import warnings
import numpy
import numpy as np
import theano.tensor.shared_randomstreams
import theano
import theano.tensor as T
import theano.tensor.signal.pool as downsample
from theano.tensor.nnet import conv
from theano import printing
from theano.tensor.shared_randomstreams import RandomStreams
import time

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

def _dropout_from_layer(rng, layer, p):
    #p is the probablity of dropping a unit
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), non_linear="tanh"):
        assert image_shape[1] == filter_shape[1]

        self.input_i = input[0]
        self.input_j = input[1]
        self.input_k = input[2]

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /numpy.prod(poolsize))

        if self.non_linear == "none" or self.non_linear == "relu":
            self.W = theano.shared(numpy.asarray(rng.uniform(low=-0.01, high=0.01, size=filter_shape),
                                                 dtype=theano.config.floatX), borrow=True, name="W_conv")
        else:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                                 dtype=theano.config.floatX), borrow=True, name="W_conv")
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name="b_conv")

        conv_out_i = conv.conv2d(input=input[0], filters=self.W, filter_shape=self.filter_shape, image_shape=self.image_shape)
        conv_out_j = conv.conv2d(input=input[1], filters=self.W, filter_shape=self.filter_shape,image_shape=self.image_shape)
        conv_out_k = conv.conv2d(input=input[2], filters=self.W, filter_shape=self.filter_shape,image_shape=self.image_shape)

        if self.non_linear == "tanh":
            conv_out_tanh_i = T.tanh(conv_out_i + self.b.dimshuffle('x', 0, 'x', 'x'))
            conv_out_tanh_j = T.tanh(conv_out_j + self.b.dimshuffle('x', 0, 'x', 'x'))
            conv_out_tanh_k = T.tanh(conv_out_k + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output_i = downsample.pool_2d(mode='max', input=conv_out_tanh_i, ds=self.poolsize, ignore_border=True)
            self.output_j = downsample.pool_2d(mode='max', input=conv_out_tanh_j, ds=self.poolsize, ignore_border=True)
            self.output_k = downsample.pool_2d(mode='max', input=conv_out_tanh_k, ds=self.poolsize, ignore_border=True)
        elif self.non_linear == "relu":
            conv_out_tanh_i = ReLU(conv_out_i + self.b.dimshuffle('x', 0, 'x', 'x'))
            conv_out_tanh_j = ReLU(conv_out_j + self.b.dimshuffle('x', 0, 'x', 'x'))
            conv_out_tanh_k = ReLU(conv_out_k + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output_i = downsample.pool_2d(mode='max', input=conv_out_tanh_i, ds=self.poolsize, ignore_border=True)
            self.output_j = downsample.pool_2d(mode='max', input=conv_out_tanh_j, ds=self.poolsize, ignore_border=True)
            self.output_k = downsample.pool_2d(mode='max', input=conv_out_tanh_k, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out_i = downsample.pool_2d(mode='max', input=conv_out_i, ds=self.poolsize, ignore_border=True)
            pooled_out_j = downsample.pool_2d(mode='max', input=conv_out_j, ds=self.poolsize, ignore_border=True)
            pooled_out_k = downsample.pool_2d(mode='max', input=conv_out_k, ds=self.poolsize, ignore_border=True)
            self.output_i = pooled_out_i + self.b.dimshuffle('x', 0, 'x', 'x')
            self.output_j = pooled_out_j + self.b.dimshuffle('x', 0, 'x', 'x')
            self.output_k = pooled_out_k + self.b.dimshuffle('x', 0, 'x', 'x')

        self.params = [self.W, self.b]

    def predict(self, new_data, batch_size):
        img_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])
        conv_out = conv.conv2d(input=new_data, filters=self.W, filter_shape=self.filter_shape, image_shape=img_shape)
        if self.non_linear == "tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = downsample.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        if self.non_linear == "relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = downsample.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = downsample.pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
            output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return output

class MLP(object):
    def __init__(self,
                 rng,
                 input1,
                 input2,
                 input3,
                 input1_t,
                 input2_t,
                 input3_t,
                 dropout_rate_v,
                 dropout_rate_t,
                 n_in,
                 n_out,
                 n2_in,
                 n2_out,
                 W1=None,
                 b1=None,
                 W2=None,
                 b2=None,
                 W1t=None,
                 b1t=None,
                 W2t=None,
                 b2t=None
                 ):
        np_rng = numpy.random.RandomState(123)
        if W1 is None:
            self.W1 = theano.shared(value=numpy.asarray(
                np_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_in + n_out)),
                    high=4 * numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX),
                    name='w1')
        else:
            self.W1 = W1
        if b1 is None:
            self.b1 = theano.shared(
                    value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                    name='b1')
        else:
            self.b1 = b1

        if W2 is None:
            self.W2 = theano.shared(value=numpy.asarray(
                np_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_in + n_out)),
                    high=4 * numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX),
                    name='w2')
        else:
            self.W2 = W2
        if b2 is None:
            self.b2 = theano.shared(
                    value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                    name='b2')
        else:
            self.b2 = b2

        if W1t is None:
            self.W1t = theano.shared(value=numpy.asarray(
                np_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n2_in + n2_out)),
                    high=4 * numpy.sqrt(6. / (n2_in + n2_out)),
                    size=(n2_in, n2_out)), dtype=theano.config.floatX),
                    name='W1t')
        else:
            self.W1t = W1t
        if b1t is None:
            self.b1t = theano.shared(
                    value=numpy.zeros((n2_out,), dtype=theano.config.floatX),
                    name='b1t')
        else:
            self.b1t = b1t

        if W2t is None:
            self.W2t = theano.shared(value=numpy.asarray(
                np_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n2_in + n2_out)),
                    high=4 * numpy.sqrt(6. / (n2_in + n2_out)),
                    size=(n2_in, n2_out)), dtype=theano.config.floatX),
                    name='W2t')
        else:
            self.W2t = W2t
        if b2t is None:
            self.b2t = theano.shared(
                    value=numpy.zeros((n2_out,), dtype=theano.config.floatX),
                    name='b2t')
        else:
            self.b2t = b2t

        self.rng = rng
        self.dropout_rate_v = dropout_rate_v
        self.dropout_rate_t = dropout_rate_t
        self.input1 = input1
        self.input2 = input2
        self.input3 = input3
        self.input1_t = input1_t
        self.input2_t = input2_t
        self.input3_t = input3_t
        self.dropout_input1 = _dropout_from_layer(self.rng, self.input1, self.dropout_rate_v)
        self.dropout_input2 = _dropout_from_layer(self.rng, self.input2, self.dropout_rate_v)
        self.dropout_input3 = _dropout_from_layer(self.rng, self.input3, self.dropout_rate_v)
        self.dropout_input1_t = _dropout_from_layer(self.rng, self.input1_t, self.dropout_rate_t)
        self.dropout_input2_t = _dropout_from_layer(self.rng, self.input2_t, self.dropout_rate_t)
        self.dropout_input3_t = _dropout_from_layer(self.rng, self.input3_t, self.dropout_rate_t)

        self.output1 = T.nnet.hard_sigmoid(T.dot(self.input1, self.W1) + T.dot(self.input1_t, self.W1t) + self.b1)
        self.output2 = T.nnet.hard_sigmoid(T.dot(self.input2, self.W2) + T.dot(self.input2_t, self.W2t) + self.b2)
        self.output3 = T.nnet.hard_sigmoid(T.dot(self.input3, self.W2) + T.dot(self.input3_t, self.W2t) + self.b2)

        self.dropout_output1 = T.nnet.hard_sigmoid(T.dot(self.dropout_input1, self.W1) + T.dot(self.dropout_input1_t, self.W1t) + self.b1)
        self.dropout_output2 =T.nnet.hard_sigmoid(T.dot(self.dropout_input2, self.W2) + T.dot(self.dropout_input2_t, self.W2t) + self.b2)
        self.dropout_output3 =T.nnet.hard_sigmoid(T.dot(self.dropout_input3, self.W2) + T.dot(self.dropout_input3_t, self.W2t) + self.b2)

        self.output1t = T.transpose(self.output1)
        self.output2t = T.transpose(self.output2)
        self.output3t = T.transpose(self.output3)

        self.dropout_output1t = T.transpose(self.dropout_output1)
        self.dropout_output2t = T.transpose(self.dropout_output2)
        self.dropout_output3t = T.transpose(self.dropout_output3)

        self.m12 = T.dot(self.output1, self.output2t).diagonal()
        self.m12 = self.m12.reshape([self.input1.shape[0], 1])
        self.m13 = T.dot(self.output1, self.output3t).diagonal()
        self.m13 = self.m13.reshape([self.input1.shape[0], 1])

        self.p_sup = T.nnet.sigmoid(self.m12 - self.m13)

        self.dropout_m12 = T.dot(self.dropout_output1, self.dropout_output2t).diagonal()
        self.dropout_m12 = self.dropout_m12.reshape([self.input1.shape[0], 1])
        self.dropout_m13 = T.dot(self.dropout_output1, self.dropout_output3t).diagonal()
        self.dropout_m13 = self.dropout_m13.reshape([self.input1.shape[0], 1])


        self.dropout_p_sup = T.nnet.sigmoid(self.dropout_m12 - self.dropout_m13)

        self.params = [self.W1, self.b1, self.W2, self.b2, self.W1t, self.W2t]
        self.sqr = (self.W1 ** 2).mean() + (self.b1 ** 2).mean() + (self.W2 ** 2).mean() + (self.b2 ** 2).mean() + \
                (self.W1t ** 2).mean() +  (self.W2t ** 2).mean()

class Rule(object):
    def __init__(self, fea_ind):
        self.fea_ind = fea_ind
        self.rule_mask = self.fea_ind[:, 0].reshape((fea_ind.shape[0], 1))

    def distr(self, w):
        distr_mij = w * self.rule_mask * (self.fea_ind[:, 1].reshape([self.fea_ind.shape[0], 1]))
        distr_mik = w * self.rule_mask * (self.fea_ind[:, 2].reshape([self.fea_ind.shape[0], 1]))
        distr_mij = distr_mij.reshape([distr_mij.shape[0], 1])
        distr_mik = distr_mik.reshape([distr_mik.shape[0], 1])
        distr = T.concatenate([distr_mij, distr_mik], axis=1)
        return distr


class LogicNN(object):
    def __init__(self,
                 input1,
                 input2,
                 input3,
                 network,
                 rules=[],
                 rule_num=3,
                 batch_size = 64,
                 n_hidden=1024,
                 attention_hidden=512,
                 C=1.0,
                 pi=0,
                 mu_param=[],
                 W1=None,
                 W2=None,
                 Wl=None,b=None,
                 w=None,c=None
                 ):
        self.input1 = input1
        self.input2 = input2
        self.input3 = input3
        self.network = network
        self.rules = rules
        self.C=C
        self.pi = theano.shared(value=pi, name='pi')
        self.mu_param = mu_param
        self.batch_size = batch_size
        self.rule_num = rule_num

        np_rng = numpy.random.RandomState(123)

        # weights of attention
        if W1 is None:
            self.W1 = theano.shared(value=numpy.asarray(
                np_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (rule_num + attention_hidden)),
                    high=4 * numpy.sqrt(6. / (rule_num + attention_hidden)),
                    size=(n_hidden, attention_hidden)), dtype=theano.config.floatX),
                    name='W1')
        else:
            self.W1 = W1

        if W2 is None:
            self.W2 = theano.shared(value=numpy.asarray(
                np_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (rule_num + attention_hidden)),
                    high=4 * numpy.sqrt(6. / (rule_num + attention_hidden)),
                    size=(n_hidden, attention_hidden)), dtype=theano.config.floatX),
                    name='W2')
        else:
            self.W2 = W2


        if Wl is None:
            self.Wl = theano.shared(value=numpy.asarray(
                np_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (rule_num + attention_hidden)),
                    high=4 * numpy.sqrt(6. / (rule_num + attention_hidden)),
                    size=(rule_num, attention_hidden)), dtype=theano.config.floatX),
                    name='Wl')
        else:
            self.Wl = Wl

        if b is None:
            self.b = theano.shared(
                    value=numpy.zeros((attention_hidden,), dtype=theano.config.floatX),
                    name='b')
        else:
            self.b = b

        if w is None:
            self.w = theano.shared(value=numpy.asarray(
                np_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (rule_num + attention_hidden)),
                    high=4 * numpy.sqrt(6. / (rule_num + attention_hidden)),
                    size=(attention_hidden, 1)), dtype=theano.config.floatX),
                    name='w')
        else:
            self.w = w

        if c is None:
            self.c = theano.shared(
                    value=numpy.zeros((rule_num,), dtype=theano.config.floatX),
                    name='c')
        else:
            self.c = c

        # build rule mask
        masks = self.rules[0].rule_mask
        for i in range(1, rule_num):
            masks = T.concatenate([masks, self.rules[i].rule_mask], axis=1)
        self.masks = masks.reshape([input1.shape[0], rule_num])
        result = []
        for i in range(0, rule_num):
            rl = np.zeros((batch_size, rule_num), dtype=theano.config.floatX)
            rl[:, i] = 1.0
            rl = theano.shared(value=rl, name='rl')
            lamb = T.dot((T.nnet.sigmoid(
                T.dot(self.network.output1, self.W1) + T.dot(self.network.output2, self.W2) +
                T.dot(self.network.output3, self.W2) + T.dot(rl, self.Wl) + self.b)), self.w) + self.c[i]
            result.append(lamb)

        lambs = result[0]
        for i in range(1, rule_num):
            lambs = T.concatenate([lambs, result[i]], axis=1)
        lambs = self.C*lambs.reshape([batch_size, rule_num])
        lambs = lambs * masks
        self.raw_rule_lambda = lambs

        temp = T.exp(lambs)*masks
        m = np.zeros([batch_size,1])
        m[:,0] = 0.0000001
        m = theano.shared(value=m, name='m')
        lambs = temp/(T.sum((temp),axis=1).reshape([input1.shape[0], 1])+m.reshape([input1.shape[0], 1]))
        self.rule_lambda = lambs
        self.rule_lambda = self.rule_lambda * masks
        self.rule_lambda.reshape((input1.shape[0], rule_num))


        # construction of q
        self.distr = self.calc_distr()
        q_m12 = self.network.m12 * 1.0
        q_m13 = self.network.m13 * 1.0
        n_q_m12 = q_m12 / (q_m12 + q_m13)
        n_q_m13 = q_m13 / (q_m12 + q_m13)
        self.q_m12 = n_q_m12 * (self.distr[:, 0].reshape((input1.shape[0], 1)))
        self.q_m13 = n_q_m13 * (self.distr[:, 1].reshape((input1.shape[0], 1)))
        self.q_sup = T.nnet.sigmoid(self.q_m12 - self.q_m13)
        self.n_q_m12_ = self.q_m12 / (self.q_m12 + self.q_m13)
        self.n_q_m13_ = self.q_m13 / (self.q_m12 + self.q_m13)
        self.q_m = T.concatenate([self.n_q_m12_, self.n_q_m13_], axis=1)

        dropout_q_m12 = self.network.dropout_m12 * 1.0
        dropout_q_m13 = self.network.dropout_m13 * 1.0
        dropout_n_q_m12 = dropout_q_m12 / (dropout_q_m12 + dropout_q_m13)
        dropout_n_q_m13 = dropout_q_m13 / (dropout_q_m12 + dropout_q_m13)
        self.dropout_q_m12 = dropout_n_q_m12 * (self.distr[:, 0].reshape((input1.shape[0], 1)))
        self.dropout_q_m13 = dropout_n_q_m13 * (self.distr[:, 1].reshape((input1.shape[0], 1)))
        self.dropout_q_sup = T.nnet.sigmoid(self.dropout_q_m12 - self.dropout_q_m13)
        dropout_n_q_m12_ = self.dropout_q_m12 / (self.dropout_q_m12 + self.dropout_q_m13)
        dropout_n_q_m13_ = self.dropout_q_m13 / (self.dropout_q_m12 + self.dropout_q_m13)
        self.dropout_q_m = T.concatenate([dropout_n_q_m12_, dropout_n_q_m13_], axis=1)
        self.params_attention = [self.W1, self.W2, self.Wl, self.b, self.w, self.c]
        self.params = self.network.params + self.params_attention 
        self.logicnn_sqr = (self.W1 ** 2).mean() + (self.W2 ** 2).mean() + (self.Wl ** 2).mean() + (self.b ** 2).mean() + (self.w ** 2).mean() + (self.c ** 2).mean()

    def calc_distr(self):
        distr_all = np.zeros([self.batch_size,2])
        distr_all = T.cast(distr_all, dtype=theano.config.floatX)
        for i,rule in enumerate(self.rules):
            distr = rule.distr(  (self.rule_lambda[:, i].reshape([self.input1.shape[0],1])))
            distr_all += distr
        return T.exp(distr_all)

    def set_pi(self, new_pi):
        self.pi.set_value(new_pi)

    def get_pi(self):
        return self.pi.get_value()

    def cost(self):
        p_m12 = self.network.m12
        p_m13 = self.network.m13

        n_p_m12 = p_m12 / (p_m12 + p_m13)
        n_p_m13 = p_m13 / (p_m12 + p_m13)

        p_m = T.concatenate([n_p_m12, n_p_m13], axis=1)
        L_sup = -T.mean(T.log(self.network.p_sup))
        L_p_q = -(T.mean(T.log(p_m)*self.q_m))
        L_sqr = self.mu_param[0]*self.network.sqr + self.mu_param[1]*self.logicnn_sqr
        cost = (1.0-self.pi)*L_sup + self.pi * L_p_q + L_sqr
        return cost, L_sup, L_p_q, L_sqr

    def dropout_cost(self):
        dropout_p_m12 = self.network.dropout_m12
        dropout_p_m13 = self.network.dropout_m13
        dropout_n_p_m12 = dropout_p_m12 / (dropout_p_m12 + dropout_p_m13)
        dropout_n_p_m13 = dropout_p_m13 / (dropout_p_m12 + dropout_p_m13)
        dropout_p_m = T.concatenate([dropout_n_p_m12, dropout_n_p_m13], axis=1)
        L_sup = -T.mean(T.log(self.network.dropout_p_sup))
        L_p_q = -(T.mean(T.log(dropout_p_m)*self.dropout_q_m))
        L_sqr = self.mu_param[0]*self.network.sqr + self.mu_param[1]*self.logicnn_sqr
        dropout_cost = (1.0 - self.pi) * L_sup + self.pi *L_p_q + L_sqr
        return dropout_cost

    def sup(self):
        q_sup = self.q_sup
        p_sup = self.network.p_sup
        return q_sup, p_sup

    def mijk(self):
        p_m12 = self.network.m12
        p_m13 = self.network.m13
        n_p_m12 = p_m12 / (p_m12 + p_m13)
        n_p_m13 = p_m13 / (p_m12 + p_m13)
        q_m12 = self.n_q_m12_
        q_m13 = self.n_q_m13_
        rule_lambda = self.rule_lambda
        masks = self.masks
        raw_rule_lambda = self.raw_rule_lambda

        return n_p_m12, n_p_m13, q_m12, q_m13, rule_lambda,masks,raw_rule_lambda