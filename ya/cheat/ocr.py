#/usr/bin/python
import numpy as np
from scipy.special import expit
from numpy import recfromcsv


def readfile(filename):
	my_data = recfromcsv(filename, delimiter=',')
	zeroes, sevens = [], []
	for data in my_data:
		if data[-1] == 0:
			zeroes.append(list(data)[:64])
		elif data[-1] == 7:
			sevens.append(list(data)[:64])
	return zeroes, sevens

def get_X(filename):
	zeroes, sevens = readfile(filename)
	print "z: ", len(zeroes), "s: ", len(sevens)
	sample = zeroes+sevens
	return sample

def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s
					
def augment_inputs(inputs):
	# @param1: input vector
	# augment 1 to the inputs matrix
	row_size = inputs.shape[0]
	ones = np.matrix(np.ones(row_size, dtype=np.int))
	inp = np.concatenate((ones.T, inputs), axis=1)
	return inp

def get_netweight(weightT, inputs):
	# @param1 :transpose of weight matrix
	# @param2 :input vector
	print inputs.shape, weightT.T.shape
	nw = inputs*weightT.T
	print "net weight"
	print nw
	return nw

def get_outputvalue(netweight, actfn=False):
	# @param1 :netweight vector
	# @param2 :activation function is used then use
	# sigmoid function
	if actfn:
		return sigmoid(netweight) # needs to be corrected
	return sigmoid(netweight)

def feedforward(weightT1, weightT2, inputs):
	# returns final output of the network
	aug_inputs= augment_inputs(inputs)
	print aug_inputs
	nw = get_netweight(weightT1, aug_inputs)
	outputs = get_outputvalue(nw, True)
	print "outputs"
	print outputs

	aug_inputs= augment_inputs(outputs)
	nw = get_netweight(weightT2, aug_inputs)
	op = get_outputvalue(nw, True)
	
	return op


# Drivers

# wT1 = np.matrix(((1, 2), (2, 3), (1, 1)))
# wT2 = np.matrix(((1, 2, 4)))
# X = np.matrix([[4, 2], [8, 1]])
# print get_netweight(wT1, X)
# print feedforward(wT1, wT2, X, 3)

x = np.matrix(get_X('optdigits.tra'))
wt1 = (1.0/100)*np.matrix(np.random.rand(10, 65))
wt2 = (1.0/100)*np.matrix(np.random.rand(2, 11))
print x
print "wt1"
print wt1
print "wt2"
print wt2
op = feedforward(wt1, wt2, x)
for i in op:
	print i

# row_size = x.shape[0]
# print row_size
# ones = np.matrix(np.ones(row_size, dtype=np.int))
# x = np.c_[ones, x]
# print x