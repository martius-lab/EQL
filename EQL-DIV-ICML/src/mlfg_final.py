"""
Multilayer function graph for system identification.
 This is able to learn typical algebraic expressions with
 maximal multiplicative/application term length given by the number of layers.
 We use regression with square error and
 L1 norm on weights to get a sparse representations.

 It follows the multilayer perceptron style, but has more complicated
 nodes.

.. math:: Each layer is

		y(x) = {f^{(1)}(W^{(1)} x),  f^{(2)}(W^{(2)} x), .., f^{(k)}(W^{(k)} x), g^{(1)}(W^{(k+1)}x, W^{(k+2)}x) }

We groups the weight matrices W1-Wk etc.
The final layer contains a division

"""
import time
import sys
import timeit
import getopt
import random
import numpy

import theano.tensor as T
from theano import In
from theano.ifelse import ifelse
import lasagne.updates as Lupdates
# if problems with importing
#   http://stackoverflow.com/questions/36088609/python-lasagne-importerror-cannot-import-batchnormlayer
from collections import OrderedDict

from utils import *

#from theano import config
#config.floatX = 'float64'

__docformat__ = 'restructedtext en'
class DivisionRegression(object):
	"""Regression layer (linear regression with division (numerator/denomator)
	"""

	def __init__(self, rng, inp, n_in, n_out, div_thresh, W=None, b=None):
		""" Initialize the parameters

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type inp: theano.tensor.TensorType
		:param inp: symbolic variable that describes the input of the
									architecture (one minibatch)

		:type n_in: int
		:param n_in: number of input units, the dimension of the space in
								 which the datapoints lie

		:type n_out: int
		:param n_out: number of output units, the dimension of the space in
									which the labels/outputs lie

		:type div_thresh: T.scalar
		:param div_thresh: threshold variable for the "soft" devision

		"""

		# attention: Formula : x*W + b where x is a row vector

		if W is None:
			# initialize with random weights W as a matrix of shape (n_in, n_out)
			W_values = numpy.asarray(
				rng.normal(loc=0, scale=numpy.sqrt(1.0 / (n_in + 2*n_out)), size=(n_in, 2*n_out)),
				dtype=theano.config.floatX
			)
			W = theano.shared(value=W_values, name='W', borrow=True)
		if b is None:
			# initialize the biases b as a vector of 2 times n_out 1s
			# (we use one to implement a more linear activation at init time)
			b = theano.shared(value=numpy.ones((2*n_out,), dtype=theano.config.floatX), name='b', borrow=True)

		self.W = W
		self.b = b

		node_inputs = T.dot(inp, self.W) + self.b
		# node_inputs is composed of input 1 and input 2 after another
		# input1 = node_inputs[0:n_out]; input2 = node_inputs[n_out:2*n_out]
		numerator = node_inputs[:, 0:n_out]
		denominator = node_inputs[:, n_out:2*n_out]
		self.output = self.activation(denominator, div_thresh) * numerator
		# parameters of the model
		self.params = [self.W, self.b]

		# keep track of model input
		self.input = inp

		self.L1 = abs(self.W).sum() + 0.01*abs(self.b).sum()
		self.L2_sqr = T.sum(self.W ** 2) + 0.01*T.sum(self.b**2)
		self.penalty = T.sum((div_thresh - denominator)*(denominator < div_thresh))
		self.extrapol_loss =  T.sum((abs(self.output)-10)*(abs(self.output)>10) + (div_thresh - denominator)*(denominator < div_thresh))

	def activation(self,x,thresh):
		return T.switch(x < thresh, 0.0, 1.0/x )

	def get_params(self):
		param_fun = theano.function(inputs=[], outputs=self.params)
		return [np.asarray(p) for p in param_fun()]

	def set_params(self, newParams):
		newb = T.vector('newb')
		newW = T.matrix('newW')
		param_fun = theano.function(inputs=[newW, newb], outputs=None, updates=[(self.W, newW), (self.b, newb)])
		return param_fun(newParams[0], newParams[1])

	def get_state(self):
		return self.get_params()

	def set_state(self, newState):
		self.set_params(newState)

	def get_weights(self):
		w_fun = theano.function(inputs=[], outputs=self.W)
		return w_fun()

	def set_out_weights(self, row, vec):  # (row)
		r = T.iscalar('row')
		new = T.vector('new')
		up_fun = theano.function(inputs=[r, new], outputs=self.W, updates=[(self.W, T.set_subtensor(self.W[r, :], new))])
		up_fun(row, vec)

	def loss(self, y):
		"""Return the mean square error of the prediction.

		:type y: theano.tensor.TensorType
		:param y: corresponds to a vector that gives for each example the
							correct value
		"""
		return T.mean(T.sqr(self.output - y))


class FGLayer(object):
	def __init__(self, rng, inp, n_in, n_per_base, layer_idx,
							 basefuncs1=None, basefuncs2=None, W=None, b=None):
		"""
		Hidden layer of Multi layer function graph: units are fully-connected and have
		the functions given by basefunc1 (arity 1) and basefunc2 (arity 2).
		Weight matrix W is of shape (n_in+1,#f1*n_per_base+2*#f2*n_per_base),
		where #f1=size(basefunc1), #f2=size(basefunc2)

		output is computed as: basefunc1[i](dot(input,W))

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type inp: theano.tensor.matrix
		:param inp: a symbolic tensor of shape (n_examples, n_in)

		:type n_in: int
		:param n_in: dimensionality of input

		:type n_per_base: int
		:param n_per_base: number of nodes per basis function

		:type basefuncs1: [int]
		:param basefuncs1: index of base functions of arity 1 to use (may contain dupplicates)
			 (list: [sin cos logistic identity])

		:type basefuncs2: [int]
		:param basefuncs2: index of base functions to arity 2 to use (may contain dupplicates)
			 (list: [mult condition])
		"""

		#TODO: get rid of rectlin and div2

		if basefuncs1 is None:
			basefuncs1 = [0, 1, 2]
		if basefuncs2 is None:
			basefuncs2 = [0]
		self.basefuncs1 = basefuncs1
		self.basefuncs2 = basefuncs2
		self.basefuncs1_uniq = list(set(basefuncs1))
		self.n_basefuncs1_uniq = len(self.basefuncs1_uniq)
		self.n_per_base = n_per_base
		self.funcs1 = ['id', 'sin', 'cos']
		self.funcs2 = ['mult']
		self.layer_idx = layer_idx

		self.input = inp
		self.n_base1 = len(basefuncs1)
		self.n_base2 = len(basefuncs2)
		n_out = (self.n_base1 + self.n_base2) * n_per_base
		n_w_out = (self.n_base1 + 2 * self.n_base2) * n_per_base
		self.n_out = n_out

		# attention: Formula : g(x*W + b) where x is a row vector
		# `W` is initialized with `W_values` which is uniformely sampled
		# from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
		# May need other values here.
		if W is None:
			W_values = numpy.asarray(
				#rng.uniform(low=-numpy.sqrt(1. / (n_in + n_w_out)), high=numpy.sqrt(1. / (n_in + n_w_out)),
				#            size=(n_in, n_w_out)),
				rng.normal(loc=0, scale=numpy.sqrt(1.0 / (n_in + n_w_out)), size=(n_in, n_w_out)),
				dtype=theano.config.floatX
			)
			W = theano.shared(value=W_values, name='W', borrow=True)
		if b is None:
			b_values = numpy.zeros((n_w_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)
		self.W = W
		self.b = b

		node_inputs = T.dot(inp, self.W) + self.b
		# node_inputs.reshape((notes.shape[0],n_base1+2*n_base2,n_per_base))
		z = node_inputs[:, :n_per_base * self.n_base1]
		z1 = node_inputs[:, n_per_base * self.n_base1:n_per_base * (self.n_base1 + self.n_base2)]
		z2 = node_inputs[:, n_per_base * (self.n_base1 + self.n_base2):]
		node_type1_values = numpy.asarray(numpy.repeat(basefuncs1, n_per_base), dtype=np.int32)
		self.nodes_type1 = theano.shared(value=node_type1_values,name='node_type1', borrow=False)
		node_type2_values = numpy.asarray(numpy.repeat(basefuncs2, n_per_base), dtype=np.int32)
		self.nodes_type2 = theano.shared(value=node_type2_values,name='node_type2', borrow=False)

		fun1 = T.switch(T.eq(self.nodes_type1, 0), z,  # identity
										T.switch(T.eq(self.nodes_type1, 1), T.sin(z),  # sine
															T.cos(z)))  # cosine
		# further functions could be maxout, sqrt, exp?

		fun2 = T.switch(T.eq(self.nodes_type2, 0), z1 * z2,  # multiplication
										# T.switch(T.eq(self.note_type2,1), z2 / (1 + T.exp(-z1)), # condition (does not work)
										z1)

		# StepOp(0.1)(z1) * z2, # if z1<0 then z2 else 0
		self.output = T.concatenate([fun1, fun2], axis=1)
		# parameters of the model
		self.params = [self.W, self.b]

		self.L1 = abs(self.W).sum() + 0.01*abs(self.b).sum()
		self.L2_sqr = T.sum(self.W ** 2) + 0.01*T.sum(self.b**2)

	def get_params(self):
		fun = theano.function(inputs=[], outputs=self.params)
		return [np.asarray(p) for p in fun()]

	def set_params(self, newParams):
		self.W.set_value(newParams[0])
		self.b.set_value(newParams[1])

	def get_state(self):
		# fun=theano.function(inputs=[],outputs=[self.nodes_type1,self.nodes_type2])
		return self.get_params() + [self.nodes_type1.get_value(), self.nodes_type2.get_value()]

	def set_state(self, newState):
		self.set_params(newState)
		if len(newState) > 2:
			self.nodes_type1.set_value(newState[2])
			self.nodes_type2.set_value(newState[3])
		else:
			print "Not full reload: missing node-types"

	def get_n_type1(self):
		return self.n_base1 * self.n_per_base

	def get_n_type2(self):
		return self.n_base2 * self.n_per_base

	def get_weights(self):
		# w_fun=theano.function(inputs=[],outputs=self.W)
		return self.W.get_value()

	def get_in_weights(self, idx):  # (column)
		node_idx = T.iscalar('node-idx')
		w_fun = theano.function(inputs=[node_idx], outputs=self.W[:, node_idx])
		return w_fun(idx)

	def set_out_weights(self, row, vec):  # (row)
		r = T.iscalar('row')
		new = T.vector('new')
		up_fun = theano.function(inputs=[r, new], outputs=self.W, updates=[(self.W, T.set_subtensor(self.W[r, :], new))])
		up_fun(row, vec)

	def set_in_weights(self, col, vec):  # (col)
		c = T.iscalar('col')
		new = T.vector('new')
		up_fun = theano.function(inputs=[c, new], outputs=self.W, updates=[(self.W, T.set_subtensor(self.W[:, c], new))])
		up_fun(col, vec)

	def get_bias(self, idx):
		node_idx = T.iscalar('node-idx')
		w_fun = theano.function(inputs=[node_idx], outputs=self.b[node_idx])
		return w_fun(idx)

	def set_bias(self, idx, value):
		node_idx = T.iscalar('node-idx')
		new = T.scalar('new')
		up_fun = theano.function(inputs=[node_idx, new], outputs=None,
														 updates=[(self.b, T.set_subtensor(self.b[node_idx], new))])
		up_fun(idx, value)

	def get_nodes_type1(self):
		# n_fun=theano.function(inputs=[], outputs=self.nodes_type1)
		return self.nodes_type1.get_value()

	def get_nodes_type2(self):
		# n_fun=theano.function(inputs=[], outputs=self.nodes_type2)
		return self.nodes_type2.get_value()

	def get_node_type1(self, idx):
		n_fun = theano.function(inputs=[], outputs=self.nodes_type1[idx])
		return n_fun()

	def set_node_type1(self, idx, typ):
		node_idx = T.iscalar('node-idx')
		new = T.iscalar('new')
		up_fun = theano.function(inputs=[node_idx, new], outputs=None,
														 updates=[(self.nodes_type1, T.set_subtensor(self.nodes_type1[node_idx], new))])
		up_fun(idx, typ)

	def getNodeFunctions(self, withnumbers=True):
		def name(func, idx):
			if withnumbers:
				return func + '-' + str(self.layer_idx) + '-' + str(idx)
			else:
				return func

		return [name(self.funcs1[bf], i) for (i, bf) in
						zip(range(1, len(self.get_nodes_type1()) + 1), self.get_nodes_type1())] + \
					 [name(self.funcs2[bf], i) for (i, bf) in
						zip(range(1, len(self.get_nodes_type2()) + 1), self.get_nodes_type2())]

	def getWeightCorrespondence(self):
		def name(func, idx):
			return func + '-' + str(self.layer_idx) + '-' + str(idx)

		return [name(self.funcs1[bf], i) for (i, bf) in
						zip(range(1, len(self.get_nodes_type1()) + 1), self.get_nodes_type1())] + \
					 [name(self.funcs2[bf], i) + ':' + '1' for (i, bf) in
						zip(range(1, len(self.get_nodes_type2()) + 1), self.get_nodes_type2())] + \
					 [name(self.funcs2[bf], i) + ':' + '2' for (i, bf) in
						zip(range(1, len(self.get_nodes_type2()) + 1), self.get_nodes_type2())]


class MLFG(object):
	"""Multi-Layer Function Graph aka EQLDiv

	A multilayer function graph, like a artificial neural network model
	that has one or more layers with hidden units of various activation functions.
	"""

	def __init__(self, rng, n_in, n_per_base, n_out, n_layer=1,
							 basefuncs1=None, basefuncs2=None, gradient=None, with_shortcuts=False):
		"""Initialize the parameters for the multilayer function graph

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type n_in: int
		:param n_in: number of input units, the dimension of the space in
		which the datapoints lie

		:type n_layer: int
		:param n_layer: number of hidden layers

		:type n_per_base: int
		:param n_per_base: number of nodes per basis function see FGLayer

		:type n_out: int
		:param n_out: number of output units, the dimension of the space in
		which the labels lie

		:type basefuncs1: [int]
		:param basefuncs1: see FGLayer

		:type basefuncs2: [int]
		:param basefuncs2: see FGLayer

		:type gradient: string
		:param gradient: type of gradient descent algo (None=="sgd+","adagrad","adadelta","nag")

		:type with_shortcuts: bool
		:param with_shortcuts: whether to use shortcut connections (output is connected to all units)

		"""
		self.input = T.matrix('input')  # the data is presented as vector input
		self.labels = T.matrix('labels')  # the labels are presented as vector of continous values
		self.rng = rng
		self.n_layers = n_layer
		self.hidden_layers = []
		self.params = []
		self.n_in = n_in
		self.n_out = n_out
		self.with_shortcuts = with_shortcuts
		self.fixL0=False

		for l in xrange(n_layer):
			if l == 0:
				layer_input = self.input
				n_input = n_in
			else:
				layer_input = self.hidden_layers[l - 1].output
				n_input = self.hidden_layers[l - 1].n_out

			hiddenLayer = FGLayer(
				rng=rng,
				inp=layer_input,
				n_in=n_input,
				n_per_base=n_per_base,
				basefuncs1=basefuncs1,
				basefuncs2=basefuncs2,
				layer_idx=l,
			)
			self.hidden_layers.append(hiddenLayer)
			self.params.extend(hiddenLayer.params)

		div_thresh = T.scalar("div_thresh")

		# The linear output layer, either it gets as input the output of ALL previous layers
		if self.with_shortcuts:
			output_layer_inp = T.concatenate([l.output for l in reversed(self.hidden_layers)], axis=1)
			output_layer_n_in = sum([l.n_out for l in self.hidden_layers])
		else:  # or just of the last hidden layer
			output_layer_inp = self.hidden_layers[-1].output
			output_layer_n_in = self.hidden_layers[-1].n_out
		self.output_layer = DivisionRegression(
			rng=rng,
			inp=output_layer_inp,
			n_in=output_layer_n_in,
			n_out=n_out,
			div_thresh=div_thresh
		)

		self.params.extend(self.output_layer.params)

		self.evalfun = theano.function(inputs=[self.input, In(div_thresh, value=0.0001)], outputs=self.output_layer.output)

		L1_reg = T.scalar('L1_reg')
		L2_reg = T.scalar('L2_reg')
		fixL0  = T.bscalar('fixL0')
		self.L1 = self.output_layer.L1 + sum([l.L1 for l in self.hidden_layers])
		self.L2_sqr = self.output_layer.L2_sqr + sum([l.L2_sqr for l in self.hidden_layers])
		self.penalty = self.output_layer.penalty


		self.loss = self.output_layer.loss
		self.errors = self.loss
		self.cost = (self.loss(self.labels) + L1_reg * self.L1 + L2_reg * self.L2_sqr + self.penalty)

		#Extrapol penalty
		self.extrapol_cost = self.output_layer.extrapol_loss

		learning_rate = T.scalar('learning_rate')

		def process_updates(par, newp):
			# print par.name
			if par.name == "W":
				# if fixL0 is True, then keep small weights at 0
				return par, ifelse(fixL0, T.switch(T.abs_(par) < 0.001, par*0, newp), newp)
			return par, newp

		print "Gradient:", gradient
		update = None
		if gradient=='sgd+' or gradient=='sgd' or gradient==None:
			gparams = [T.grad(self.cost, param) for param in self.params]
			update = OrderedDict([(param, param - (learning_rate * gparam).clip(-1.0, 1.0))
								 for param, gparam in zip(self.params, gparams)])
		elif gradient=='adam':
			update = Lupdates.adam(self.cost, self.params, learning_rate, epsilon=1e-04)
		elif gradient == 'adadelta':
			update = Lupdates.adadelta(self.cost, self.params,learning_rate)
		elif gradient == 'rmsprop':
			update = Lupdates.rmsprop(self.cost, self.params,learning_rate)
		elif gradient == 'nag':
			update = Lupdates.nesterov_momentum(self.cost,self.params,learning_rate)
		else:
			assert("unknown gradient " + gradient)

		#Extrapol sanity gradient computation:

		extrapol_updates = Lupdates.adam(self.extrapol_cost, self.params, learning_rate, epsilon=1e-04)

		updates = [process_updates(*up) for up in update.items()]
		self.train_model = theano.function(
			inputs=[self.input, self.labels, L1_reg, L2_reg, fixL0, learning_rate, div_thresh],
			outputs=self.cost,
			updates=updates,
		)
		# avoid too large outputs in extrapolation domain
		self.remove_extrapol_error = theano.function(
			inputs=[self.input, learning_rate, div_thresh],
			outputs=self.extrapol_cost,
			updates=extrapol_updates,
		)

		self.test_model = theano.function(
			inputs=[self.input, self.labels, In(div_thresh, value=0.0001)],
			outputs=self.errors(self.labels),
		)
		self.validate_model = theano.function(
			inputs=[self.input, self.labels, In(div_thresh, value=0.0001)],
			outputs=self.errors(self.labels),
		)
		self.L1_loss = theano.function(
			inputs=[],
			outputs=self.L1,
		)
		self.MSE = theano.function(
			inputs=[self.input, self.labels, In(div_thresh, value=0.0001)],
			outputs=self.errors(self.labels),
		)

	@staticmethod
	def vec_norm(vec):
		return T.sqrt(T.sum(T.sqr(vec)))

	@staticmethod
	def vec_normalize(vec):
		norm = MLFG.vec_norm(vec)
		return vec / (norm + 1e-10)

	def get_params(self):
		paramfun = theano.function(inputs=[], outputs=self.params)
		return paramfun()

	def get_state(self):
		return [l.get_state() for l in self.hidden_layers] + [self.output_layer.get_state()]

	def set_state(self, newState):
		for (s, l) in zip(newState, self.hidden_layers + [self.output_layer]):
			l.set_state(s)

	def evaluate(self, input):
		return self.evalfun(cast_to_floatX(input))

	def get_n_units_type1(self):
		return sum([l.get_n_type1() for l in self.hidden_layers])

	def get_n_units_type2(self):
		return sum([l.get_n_type2() for l in self.hidden_layers])

	# sparsity
	def get_num_active_units(self, thresh=0.1):
		# count units with nonzero input * output weights (only non-identity units)
		# in principle one could make a backward scan and identify units without path to the output, but
		# we keep it simpler.
		total = 0
		for layer_idx in range(0, self.n_layers):
			layer = self.hidden_layers[layer_idx]
			in_weights = layer.get_weights()
			#bias = layer.get_biasForNumActive()
			out_weights = self.hidden_layers[layer_idx + 1].get_weights() if layer_idx + 1 < self.n_layers \
				else self.output_layer.get_weights()
			# noinspection PyTypeChecker
			in_weight_norm = np.linalg.norm(in_weights, axis=0, ord=1)
			out_weight_norm = np.linalg.norm(out_weights, axis=1, ord=1)

			# countering non-identity unary units
			# noinspection PyTypeChecker
			for i in range(layer.get_n_type1()):
				if (out_weight_norm[i]*in_weight_norm[i] > thresh*thresh and layer.get_nodes_type1()[i] != 0): #nodes_type1 matrix of 00...011...1x`
					total += 1
					#print layer_idx, layer.get_nodes_type1()[i], out_weight_norm[i], in_weight_norm[i]
			# Note that multiplication units can also be linear units of one of their inputs is constant
			#  here the norm of the input weight is set to 0 if onw of the inputs is below thresh
			for i in range(layer.get_n_type2()):
				if (in_weight_norm[layer.get_n_type1() + i] > thresh and \
					in_weight_norm[layer.get_n_type1() + layer.get_n_type2() + i]  > thresh):
					in_weight_norm[layer.get_n_type1() + i] += in_weight_norm[layer.get_n_type1() + layer.get_n_type2() + i]
				else:
					in_weight_norm[layer.get_n_type1() + i] = 0

			# countering non-identity multiplicative units
			for i in range(layer.get_n_type1(), layer.get_n_type1() + layer.get_n_type2()):
				if (out_weight_norm[i]*in_weight_norm[i] > thresh*thresh):
					total += 1
					#print layer_idx, "mult", out_weight_norm[i], in_weight_norm[i]

		return total


	def get_active_units_old(self, thresh=0.05):
		# quick hack: count units with nonzero output weights not counting the inputs
		total = 0
		for layer_idx in range(1, self.n_layers + 1):
			layer = self.hidden_layers[layer_idx] if layer_idx < self.n_layers else self.output_layer
			# noinspection PyTypeChecker
			out_weight_norm = np.linalg.norm(layer.get_weights(), axis=1, ord=1)
			total += sum(out_weight_norm > thresh)
		return total


def test_mlfg(datasets, learning_rate=0.01, L1_reg=0.001, L2_reg=0.00, n_epochs=200,
							batch_size=20, n_layer=1, n_per_base=5, basefuncs1=None, basefuncs2=None,
							with_shortcuts=False, id=None,
							classifier=None,
							gradient=None,
							init_state=None,
							verbose=True, param_store=None,
							reg_start=0, reg_end=None,
							validate_every=50,
							k=100
							):
	"""
	:type datasets: ((matrix,matrix),(matrix,matrix),(matrix,matrix))
	:param datasets: ((train-x,train-y),(valid-x,valid-y),(test-x,test-y))

	:type learning_rate: float
	:param learning_rate: learning rate used (factor for the stochastic
	gradient

	:type L1_reg: float
	:param L1_reg: L1-norm's weight when added to the cost (see
	regularization)

	:type L2_reg: float
	:param L2_reg: L2-norm's weight when added to the cost (see
	regularization)

	:type n_epochs: int
	:param n_epochs: maximal number of epochs to run the optimizer

	:type param_store: []
	:param param_store: if not None then the weights of each episode are stored here

	:type id: int
	:param id: id of run (also used as random seed. if None then the time is used as seed

	:param init_state: initial state for classifier to use

 """
	train_set_x, train_set_y = cast_dataset_to_floatX(datasets[0])
	valid_set_x, valid_set_y = cast_dataset_to_floatX(datasets[1])
	MAX_INPUT_VAL = np.max(abs(train_set_x))
	print "Max input value is: ", MAX_INPUT_VAL
	#extra_set_x, extra_set_y = cast_dataset_to_floatX(extrapol_dataset[0]) #0 has to be used and extrapol_dataset[1] has null entry
	#extra_set_x has dimensions 5000x4 for cp_new dataset ... verified by the following print statement

	#print "extrapol dim: ", len(extra_set_x), len(extra_set_x[0]), len(extra_set_y), len(extra_set_y[0])

	if len(datasets) > 2:
		test_set_x, test_set_y = cast_dataset_to_floatX(datasets[2])
		n_test_batches = test_set_x.shape[0] / batch_size
	else:
		test_set_x = test_set_y = None
		n_test_batches = 0
	n_train_batches = train_set_x.shape[0] / batch_size
	n_valid_batches = valid_set_x.shape[0] / batch_size

	inputdim = len(datasets[0][0][0])
	outputdim = len(datasets[0][1][0])
	if verbose: print "Input/output dim:", (inputdim, outputdim)
	if verbose: print "Training set, test set:", (train_set_x.shape[0], test_set_x.shape[0])

	######################
	# BUILD ACTUAL MODEL #
	######################
	print '... building the model'

	rng = numpy.random.RandomState(int(time.time()) if id is None else id)
	if classifier is None:
		classifier = MLFG(
			rng=rng,
			n_in=inputdim,
			n_per_base=n_per_base,
			n_out=outputdim,
			n_layer=n_layer,
			gradient=gradient,
			basefuncs1=basefuncs1,
			basefuncs2=basefuncs2,
			with_shortcuts=with_shortcuts,
		)
	if init_state:
		classifier.set_state(init_state)

	###############
	# TRAIN MODEL #
	###############
	print '... training'
	sys.stdout.flush()

	# early-stopping parameters
	improvement_threshold = 0.99  # a relative improvement of this much is considered significant

	best_validation_error = numpy.inf
	this_validation_error = numpy.inf
	best_epoch = 0
	test_score = 0.
	best_state = classifier.get_state()

	start_time = timeit.default_timer()

	epoch = 0
	done_looping = False
	train_errors = []
	extrapol_train_errors = []
	validation_errors = []
	test_errors = []
	MSE = []
	L1 = []

	if param_store is not None:
		param_store.append(classifier.get_params())

	while (epoch < n_epochs) and (not done_looping):
		#print epoch #remove
		special_penalty = 0
		epoch = epoch + 1
		reg_factor = 0.0
		if reg_start < epoch <= reg_end:
			reg_factor = 1.0
			L1.append([epoch, np.asscalar(classifier.L1_loss())])
			if (epoch - reg_start)%k == 0 and epoch < reg_end:
				special_penalty = 1

		temp = zip(list(train_set_x),list(train_set_y))
		random.shuffle(temp)
		train_set_x, train_set_y = zip(*temp)
		train_set_x = numpy.asarray(train_set_x)
		train_set_y = numpy.asarray(train_set_y)
		del temp[:]
		minibatch_avg_cost = 0.0
		for minibatch_index in xrange(n_train_batches):
			index = minibatch_index
			minibatch_avg_cost += classifier.train_model(
				input=train_set_x[index * batch_size: (index + 1) * batch_size],
				labels=train_set_y[index * batch_size: (index + 1) * batch_size],
				L1_reg=L1_reg * reg_factor,
				L2_reg=L2_reg * reg_factor,
				fixL0 = epoch > reg_end,
				div_thresh = 1.0/np.sqrt(epoch + 1),
				learning_rate=learning_rate,
			)


		if special_penalty == 1:
			#max input val would ensure we don't have poles anywhere in twice the interpolation region
			n_num, n_in = train_set_x.shape
			extra_set_x = (2*np.random.rand(n_num, n_in)-1.0)*MAX_INPUT_VAL
			assert extra_set_x.shape == train_set_x.shape

			for x in range(n_num):
				for y in range(n_in):
					if (extra_set_x[x][y] >=0.0):
						extra_set_x[x][y] += MAX_INPUT_VAL
					else:
						extra_set_x[x][y] -= MAX_INPUT_VAL

			extrapol_error_training = 0.0

			for minibatch_index in xrange(n_train_batches):
				index = minibatch_index
				extrapol_error_training += classifier.remove_extrapol_error(
											input=extra_set_x[index * batch_size: (index + 1) * batch_size],
											div_thresh = 1.0/np.sqrt(epoch + 1),
											learning_rate=learning_rate,
											)
			extrapol_train_errors.append([epoch, extrapol_error_training/n_train_batches])

		train_errors.append([epoch, minibatch_avg_cost/n_train_batches])

		if param_store is not None:
			param_store.append(classifier.get_params())

		if epoch == 1 or epoch % validate_every == 0 or epoch == n_epochs:
			this_validation_errors = [classifier.validate_model(
				input=valid_set_x[index * batch_size:(index + 1) * batch_size],
				labels=valid_set_y[index * batch_size:(index + 1) * batch_size])
																for index in xrange(n_valid_batches)]
			this_validation_error = np.asscalar(numpy.mean(this_validation_errors))

			validation_errors.append([epoch, this_validation_error])

			this_MSE = [classifier.MSE(input=train_set_x[index*batch_size:(index + 1)*batch_size],
						labels=train_set_y[index*batch_size: (index + 1)*batch_size]) for index in xrange(n_train_batches)]
			MSE.append([epoch, np.asscalar(np.mean(this_MSE))])

			if verbose:
				print(
					'epoch %i, minibatch %i/%i, minibatch_avg_cost %f validation error %f' %
					(
						epoch,
						minibatch_index + 1,
						n_train_batches,
						minibatch_avg_cost,
						this_validation_error
					)
				)

			# test it on the test set
			if test_set_x is not None:
				test_losses = [classifier.test_model(
					input=test_set_x[index * batch_size:(index + 1) * batch_size],
					labels=test_set_y[index * batch_size:(index + 1) * batch_size])
											 for index in xrange(n_test_batches)]
				this_test_score = np.asscalar(numpy.mean(test_losses))
				test_errors.append([epoch, this_test_score])
			else:
				this_test_score = np.inf

			# if we got the best validation score until now
			if this_validation_error < best_validation_error:
				if this_validation_error < best_validation_error * improvement_threshold:
					best_state = classifier.get_state()

				best_validation_error = this_validation_error
				best_epoch = epoch
				test_score = this_test_score

				if verbose:
					print(('epoch %i, minibatch %i/%i, test error of '
								 'best model %f') %
								(epoch, minibatch_index + 1, n_train_batches,
								 test_score))

		if epoch % 100 == 0:
			print "Epoch: ", epoch, "\tBest val error: ", best_validation_error, "\tcurrent val error: ", this_validation_error
			sys.stdout.flush()

	end_time = timeit.default_timer()
	time_required = (end_time - start_time) / 60.
	print(('Optimization complete. Best validation score of %f '
				 'obtained at epoch %i, with test performance %f ') %
				(best_validation_error, best_epoch + 1, test_score))
	print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % time_required)

	if verbose:
		numpy.set_printoptions(precision=4, suppress=True)
		print(classifier.get_params())
	return {'train_losses': numpy.asarray(train_errors),
			'extrapol_train_losses':numpy.asarray(extrapol_train_errors),
			'MSE':numpy.asarray(MSE),
			'L1':numpy.asarray(L1),
			'val_errors': numpy.asarray(validation_errors),
			'test_errors': numpy.asarray(test_errors),
			'classifier': classifier,
			'test_score': test_score,
			'val_score': this_validation_error,
			'best_val_score': best_validation_error,
			'best_epoch': best_epoch,
			'best_state': best_state,
			'num_active': classifier.get_num_active_units(),
			'runtime': time_required
			}


def usage():
	print(sys.argv[0] + "[-i id -d dataset -p extrapolationdataset -l layers -e epochs -n nodes -r learningrate --initfile=file --batchsize=k --l1=l1reg --l2=l2reg --shortcut --reg_start=start --reg_end=end --resfolder -v]")


if __name__ == "__main__":
	dataset_file = None
	extra_pol_test_sets = []
	extra_pols = []
	n_epochs = 1200
	n_layers = 3
	n_nodes = 10
	batch_size = 20
	init_file = None
	init_state = None
	gradient = "sgd"
	L1_reg = 0.001
	L2_reg = 0.001
	learning_rate = 0.01
	with_shortcuts = False
	reg_start = 0
	reg_end = None
	output = False
	verbose = 0
	k=99999999999
	id = np.random.randint(0, 1000000)
	result_folder = "./"
	basefuncs1 = [0, 1, 2]
	iterNum=0


	theano.gof.compilelock.set_lock_status(False)

	try:
		opts, args = getopt.getopt(sys.argv[1:], "hv:i:d:p:l:e:n:f:co",
															 ["help", "verbose=", "id=", "dataset=", "extrapol=", "layers=", "epochs=",
																"nodes=", "l1=", "l2=", "lr=", "resfolder=",
																"batchsize=", "initfile=", "gradient=",
																"reg_start=", "reg_end=", "shortcut", "output","k_update=","iterNum="
																])
	except getopt.GetoptError:
		usage()
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			usage()
			sys.exit()
		elif opt in ("-v", "--verbose"):
			verbose = int(arg)
		elif opt in ("-i", "--id"):
			id = int(arg)
		elif opt in ("-d", "--dataset"):
			dataset_file = arg
		elif opt in ("-p", "--extrapol"):
			extra_pol_test_sets.append(arg)
		elif opt in ("-l", "--layers"):
			n_layers = int(arg)
		elif opt in ("-e", "--epochs"):
			n_epochs = int(arg)
		elif opt in ("--batchsize"):
			batch_size = int(arg)
		elif opt in ("--l1"):
			L1_reg = float(arg)
		elif opt in ("--l2"):
			L2_reg = float(arg)
		elif opt in ("--lr"):
			learning_rate = float(arg)
		elif opt in ("-n", "--nodes"):
			n_nodes = int(arg)
		elif opt in ("-c", "--shortcut"):
			with_shortcuts = True
		elif opt in ("--initfile"):
			init_file = arg
		elif opt in ("--gradient"):
			gradient= arg
		elif opt in ("--reg_start"):
			reg_start = int(arg)
		elif opt in ("--reg_end"):
			reg_end = int(arg)
		elif opt in ("-o", "--output"):
			output = True
		elif opt in ("-f", "--resfolder"):
			result_folder = arg
		elif opt in ("--iterNum"):
			iterNum = int(arg)
		elif opt in ("--k_update"):
			k = int(arg)
	# load dataset
	if not dataset_file:
		print("provide datasetfile!")
		usage()
		exit(1)
	dataset = load_data(dataset_file)

	# load extrapolation test
	if len(extra_pol_test_sets) > 0:
		if verbose > 0:
			print("do also extrapolation test(s)!")
		extra_pols = [load_data(test_set) for test_set in extra_pol_test_sets]

	if init_file:
		with open(init_file, 'rb') as f:
			init_state = cPickle.load(f)
			print "load initial state from file " + init_file


	if not os.path.exists(result_folder):
		os.makedirs(result_folder)

	name = result_folder + "/" + str(id)
	print ("Results go into " +  result_folder)

	result = test_mlfg(datasets=dataset, k=k, n_epochs=n_epochs, verbose=verbose > 0, learning_rate=learning_rate,
										 L1_reg=L1_reg, L2_reg=L2_reg, basefuncs2=[0], basefuncs1=basefuncs1, n_layer=n_layers,
										 n_per_base=n_nodes, id=id, gradient=gradient,
										 batch_size=batch_size, init_state=init_state,
										 reg_start=reg_start, reg_end=reg_end, with_shortcuts=with_shortcuts,
										 )

	classifier = result['classifier']
	with  open(name + '.best_state', 'wb') as f:
		cPickle.dump(result['best_state'], f, protocol=cPickle.HIGHEST_PROTOCOL)
	with  open(name + '.last_state', 'wb') as f:
		cPickle.dump(classifier.get_state(), f, protocol=cPickle.HIGHEST_PROTOCOL)

	extra_scores = []
	extra_scores_best = []
	for extra in extra_pols:
		extra_set_x, extra_set_y = cast_dataset_to_floatX(extra[0])
		extra_scores.append(classifier.test_model(input=extra_set_x, labels=extra_set_y))
	# also for best_state
	classifier.set_state(result['best_state'])
	for extra in extra_pols:
		extra_set_x, extra_set_y = cast_dataset_to_floatX(extra[0])
		extra_scores_best.append(classifier.test_model(input=extra_set_x, labels=extra_set_y))

	result_line = ""
	with  open(name + '.res', 'w') as f:
		f.write('#C k iter layers epochs nodes lr L1 L2 shortcut batchsize regstart regend' +
						' id dataset gradient numactive bestnumactive bestepoch runtime' +
						"".join([' extrapol' + str(i) for i in range(1, len(extra_scores) + 1)]) +
						"".join([' extrapolbest' + str(i) for i in range(1, len(extra_scores_best) + 1)]) +
						' valerror valerrorbest testerror\n')
		f.write('# extra datasets: ' + " ".join(extra_pol_test_sets) + '\n')
		result_line = [str(k), str(iterNum), str(n_layers), str(n_epochs), str(n_nodes), str(learning_rate),
									 str(L1_reg), str(L2_reg),
									 str(with_shortcuts), str(batch_size), str(reg_start), str(reg_end),
									 str(id), dataset_file, gradient,
									 str(result['num_active']), str(classifier.get_num_active_units()),
									 str(result['best_epoch']), str(result['runtime'])] + \
									[str(e) for e in extra_scores] + \
									[str(e) for e in extra_scores_best] + \
									[str(result['val_score']), str(result['best_val_score']), str(result['test_score'])]
		f.write(str.join('\t', result_line) + '\n')

	with open(name + '.validerror', 'wb') as csvfile:
		a = csv.writer(csvfile, delimiter='\t')
		a.writerows([["#C epoch", "val_error"]])
		a.writerows([["# "] + result_line])
		a.writerows(result['val_errors'])

	with open(name + '.MSE', 'wb') as csvfile:
		a = csv.writer(csvfile, delimiter='\t')
		a.writerows([["#C epoch", "MSE"]])
		a.writerows([["# "] + result_line])
		a.writerows(result['MSE'])

	with open(name + '.L1', 'wb') as csvfile:
		a = csv.writer(csvfile, delimiter='\t')
		a.writerows([["#C epoch", "L1"]])
		a.writerows([["# "] + result_line])
		a.writerows(result['L1'])

	output=1
	if output:
		with open(name + '.trainloss', 'wb') as csvfile:
			a = csv.writer(csvfile, delimiter='\t')
			a.writerows([["#C epoch", "train_loss"]])
			a.writerows([["# "] + result_line])
			a.writerows(result['train_losses'])

		if len(result['test_errors']) > 0:
			with open(name + '.testerrors', 'wb') as csvfile:
				a = csv.writer(csvfile, delimiter='\t')
				a.writerows([["#C epoch", "test_error"]])
				a.writerows([["# "] + result_line])
				a.writerows(result['test_errors'])

		with open(name + '.extrapoltrainloss', 'wb') as csvfile:
			a = csv.writer(csvfile, delimiter='\t')
			a.writerows([["#C epoch", "extrapol_train_loss"]])
			a.writerows([["# "] + result_line])
			a.writerows(result['extrapol_train_losses'])
