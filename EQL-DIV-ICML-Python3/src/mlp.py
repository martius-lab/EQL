"""
Multilayer function graph for system identification
 This will simply use regression in the square error with
 L1 norm on weights to get a sparse representation

 It follows the multilayer perceptron style, but has more complicated
 nodes.

.. math:: Each layer is

    y(x) = {f^{(1)}(W^{(1)} x),  f^{(2)}(W^{(2)} x), .., f^{(k)}(W^{(k)} x), g^{(1)}(W^{(k+1)}x, W^{(k+2)}x) }

We groups the weight matrices W1-Wk etc

"""
import time
import os
import sys
import timeit
import pickle
import getopt
import csv

import numpy as np

import utils

import theano
import theano.tensor as T
#import lasagne.updates as Lupdates
theano.config.floatX = 'float64'
__docformat__ = 'restructedtext en'


def logistic(x):
  return 1 / (1 + T.exp(-x))


class LinearRegression(object):
  """Regression layer (linear regression)
  """

  def __init__(self, rng, inp, n_in, n_out):
    """ Initialize the parameters of the linear regression

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

    """
    # initialize with random weights W as a matrix of shape (n_in, n_out)
    W_values = np.asarray(
      rng.uniform(
        low=-np.sqrt(1.0 / (n_in + n_out)),
        high=np.sqrt(1.0 / (n_in + n_out)),
        size=(n_in, n_out)
      ),
      dtype=theano.config.floatX
    )
    self.W = theano.shared(value=W_values, name='W', borrow=True)
    # initialize the biases b as a vector of n_out 0s
    self.b = theano.shared(
      value=np.zeros(
        (n_out,),
        dtype=theano.config.floatX
      ),
      name='b',
      borrow=True
    )

    self.output = T.dot(inp, self.W) + self.b

    # parameters of the model
    self.params = [self.W, self.b]

    # keep track of model input
    self.input = inp

    self.L1 = abs(self.W).sum()
    self.L2_sqr = T.sum(self.W ** 2)

  def get_params(self):
    paramfun = theano.function(inputs=[], outputs=self.params)
    return paramfun()

  def set_params(self, newParams):
    newb = T.dvector('newb')
    newW = T.dmatrix('newW')
    param_fun = theano.function(inputs=[newW, newb], outputs=None, updates=[(self.W, newW), (self.b, newb)])
    return param_fun(newParams[0], newParams[1])

  def get_weights(self):
    return self.W.get_value()

  def loss(self, y):
    """Return the mean square error of the prediction.

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
              correct value
    """
    return T.mean(T.sqr(self.output - y))


class HiddenLayer(object):
  def __init__(self, rng, inp, n_in, n_units, layer_idx, W=None, b=None):
    """
    Hidden layer of Multi layer network

    :type rng: numpy.random.RandomState
    :param rng: a random number generator used to initialize weights

    :type inp: theano.tensor.dmatrix
    :param inp: a symbolic tensor of shape (n_examples, n_in)

    :type n_in: int
    :param n_in: dimensionality of input

    :type n_units: int
    :param n_units: number of hidden nodes

    """

    self.layer_idx = layer_idx

    self.input = inp
    n_out = (n_units)
    self.n_out = n_out

    # `W` is initialized with `W_values` which is uniformely sampled
    # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
    # May need other values here.
    if W is None:
      W_values = np.asarray(
        rng.uniform(
          low=-np.sqrt(6. / (n_in + n_out)),
          high=np.sqrt(6. / (n_in + n_out)),
          size=(n_in, n_out)
        ),
        dtype=theano.config.floatX
      )
      W = theano.shared(value=W_values, name='W', borrow=True)
    if b is None:
      b_values = np.zeros((n_out,), dtype=theano.config.floatX)
      b = theano.shared(value=b_values, name='b', borrow=True)
    self.W = W
    self.b = b

    node_inputs = T.dot(inp, self.W) + self.b
    self.output = T.tanh(node_inputs)

    self.params = [self.W, self.b]

    self.L1 = abs(self.W).sum()
    self.L2_sqr = T.sum(self.W ** 2)

  def get_params(self):
    paramfun = theano.function(inputs=[], outputs=self.params)
    return paramfun()

  def set_params(self, newParams):
    newb = T.dvector('newb')
    newW = T.dmatrix('newW')
    param_fun = theano.function(inputs=[newW, newb], outputs=None, updates=[(self.W, newW), (self.b, newb)])
    return param_fun(newParams[0], newParams[1])

  def get_weights(self):
    return self.W.get_value()


class MLP(object):
  """Multi-Layer Function Graph

  A multilayer function graph, like a artificial neural network model
  that has one layer or more of hidden units and various activations.
  """

  def __init__(self, rng, n_in, n_units, n_out, n_layer=1, gradient=None):
    """Initialize the parameters for the multilayer function graph

    :type rng: numpy.random.RandomState
    :param rng: a random number generator used to initialize weights

    :type n_in: int
    :param n_in: number of input units, the dimension of the space in
    which the datapoints lie

    :type n_layer: int
    :param n_layer: number of hidden layers

    :type n_units: int
    :param n_units: number of nodes per hidden layer

    :type n_out: int
    :param n_out: number of output units, the dimension of the space in
    which the labels lie


    """
    self.input = T.matrix('input')  # the data is presented as vector input
    self.labels = T.matrix('labels')  # the labels are presented as vector of continous values

    self.n_layers = n_layer
    self.hidden_layers = []
    self.params = []
    self.n_in = n_in
    self.n_out = n_out

    for l in range(n_layer):
      if l == 0:
        layer_input = self.input
        n_input = n_in
      else:
        layer_input = self.hidden_layers[l - 1].output
        n_input = self.hidden_layers[l - 1].n_out

      hiddenLayer = HiddenLayer(
        rng=rng,
        inp=layer_input,
        n_in=n_input,
        n_units=n_units,
        layer_idx=l,
      )
      self.hidden_layers.append(hiddenLayer)
      self.params.extend(hiddenLayer.params)

    # The linear output layer gets as input the hidden units
    # of the hidden layer
    self.output_layer = LinearRegression(
      rng=rng,
      inp=self.hidden_layers[-1].output,
      n_in=self.hidden_layers[-1].n_out,
      n_out=n_out
    )
    self.params.extend(self.output_layer.params)

    self.evalfun = theano.function(inputs=[self.input], outputs=self.output_layer.output)

    L1_reg = T.dscalar('L1_reg')
    L2_reg = T.dscalar('L2_reg')
    self.L1 = self.output_layer.L1 + sum([l.L1 for l in self.hidden_layers])
    self.L2_sqr = self.output_layer.L2_sqr + sum([l.L2_sqr for l in self.hidden_layers])

    self.loss = self.output_layer.loss
    self.errors = self.loss
    self.cost = (self.loss(self.labels) + L1_reg * self.L1 + L2_reg * self.L2_sqr)

    learning_rate = T.dscalar('learning_rate')

    updates = []
    if gradient is None:
      gradient = "sgd"
    print("Gradient:", gradient)
    if gradient == 'sgd':
      gparams = [T.grad(self.cost, param) for param in self.params]
      updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(self.params, gparams)
        ]
#    elif gradient == 'adam':
#      updates = Lupdates.adam(self.cost, self.params, learning_rate)
    else:
      assert ("unknown gradient " + gradient == False)

    self.train_model = theano.function(
      inputs=[self.input, self.labels, L1_reg, L2_reg, learning_rate],
      outputs=self.cost,
      updates=updates,
    )
    self.test_model = theano.function(
      inputs=[self.input, self.labels],
      outputs=self.loss(self.labels),
    )
    self.validate_model = theano.function(
      inputs=[self.input, self.labels],
      outputs=self.errors(self.labels),
    )

  def get_params(self):
    paramfun = theano.function(inputs=[], outputs=self.params)
    return paramfun()

  def get_state(self):
    return [l.get_params() for l in self.hidden_layers] + [self.output_layer.get_params()]

  def set_state(self, newState):
    for (s, l) in zip(newState, self.hidden_layers + [self.output_layer]):
      l.set_params(s)

  def get_active_units(self, thresh=0.1):
    # count units with nonzero input * output weights
    # in principle one could make a backward scan and identify units without path to the output
    total = 0
    for layer_idx in range(0, self.n_layers):
      layer = self.hidden_layers[layer_idx]
      in_weights = layer.get_weights()
      out_weights = self.hidden_layers[layer_idx + 1].get_weights() if layer_idx + 1 < self.n_layers \
        else self.output_layer.get_weights()
      # noinspection PyTypeChecker
      in_weight_norm = np.linalg.norm(in_weights, axis=0, ord=1)
      # noinspection PyTypeChecker
      out_weight_norm = np.linalg.norm(out_weights, axis=1, ord=1)
      total += sum(
        (out_weight_norm * in_weight_norm) > thresh * thresh)
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

  def evaluate(self, input):
    return self.evalfun(input)


def test_mlp(datasets, learning_rate=0.01, L1_reg=0.001, L2_reg=0.00, n_epochs=200,
             batch_size=20, n_layer=1, n_units=30, classifier=None, init_state=None,
             gradient=None, verbose=True, param_store=None, id=None,
             validate_every=50, reg_start=0, reg_end=None
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
 """

  train_set_x, train_set_y = datasets[0]
  valid_set_x, valid_set_y = datasets[1]
  if len(datasets) > 2 and len(datasets[2]) == 2:
    test_set_x, test_set_y = datasets[2]
    n_test_batches = test_set_x.shape[0] // batch_size
  else:
    test_set_x = test_set_y = None
    n_test_batches = 0
  n_train_batches = train_set_x.shape[0] // batch_size
  n_valid_batches = valid_set_x.shape[0] // batch_size

  inputdim = len(datasets[0][0][0])
  outputdim = len(datasets[0][1][0])
  if verbose: print("Input/output dim:", (inputdim, outputdim))
  if verbose: print("Training set, test set:", (train_set_x.shape[0], test_set_x.shape[0]))

  ######################
  # BUILD ACTUAL MODEL #
  ######################
  print('... building the model')

  rng = np.random.RandomState(int(time.time()) if id is None else id)
  if classifier is None:
    classifier = MLP(
      rng=rng,
      n_in=inputdim,
      n_units=n_units,
      n_out=outputdim,
      n_layer=n_layer,
      gradient=gradient,
    )
  if init_state:
    classifier.set_state(init_state)

  ###############
  # TRAIN MODEL #
  ###############
  print('... training')
  sys.stdout.flush()

  # early-stopping parameters
  improvement_threshold = 0.99  # a relative improvement of this much is considered significant

  best_validation_error = np.inf
  this_validation_error = np.inf
  best_epoch = 0
  test_score = 0.
  best_state = classifier.get_state()

  start_time = timeit.default_timer()

  epoch = 0
  done_looping = False
  train_errors = []
  validation_errors = []
  test_errors = []

  if param_store is not None:
    param_store.append(classifier.get_params())

  while (epoch < n_epochs) and (not done_looping):
    epoch += 1
    reg_factor = 0.0
    if reg_start < epoch <= reg_end:
      reg_factor = 1.0

    for minibatch_index in range(n_train_batches):
      index = minibatch_index
      minibatch_avg_cost = classifier.train_model(
        input=train_set_x[index * batch_size: (index + 1) * batch_size],
        labels=train_set_y[index * batch_size: (index + 1) * batch_size],
        L1_reg=L1_reg * reg_factor,
        L2_reg=L2_reg * reg_factor,
        learning_rate=learning_rate,
      )
      # if verbose:
      #    print('epoch %i, minibatch %i cost: %f' %(epoch, minibatch_index, minibatch_avg_cost))
      # if(minibatch_avg_cost>2):
      #    np.set_printoptions(precision=4,suppress=True)
      #    print(classifier.get_params())

    train_errors.append([epoch, minibatch_avg_cost])

    if param_store is not None:
      param_store.append(classifier.get_params())

    # perform validation
    if epoch == 1 or epoch % validate_every == 0 or epoch == n_epochs:
      this_validation_errors = [classifier.validate_model(
        input=valid_set_x[index * batch_size:(index + 1) * batch_size],
        labels=valid_set_y[index * batch_size:(index + 1) * batch_size])
                                for index in range(n_valid_batches)]
      this_validation_error = np.asscalar(np.mean(this_validation_errors))

      validation_errors.append([epoch, this_validation_error])

      if verbose:
        print((
          'epoch %i, minibatch %i/%i, minibatch_avg_cost %f validation error %f' %
          (
            epoch,
            minibatch_index + 1,
            n_train_batches,
            minibatch_avg_cost,
            this_validation_error
          )
        ))

      # test it on the test set
      if test_set_x is not None:
        test_losses = [classifier.test_model(
          input=test_set_x[index * batch_size:(index + 1) * batch_size],
          labels=test_set_y[index * batch_size:(index + 1) * batch_size])
                       for index in range(n_test_batches)]
        this_test_score = np.asscalar(np.mean(test_losses))
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
          print((('     epoch %i, minibatch %i/%i, test error of '
                 'best model %f') %
                (epoch, minibatch_index + 1, n_train_batches,
                 test_score)))

    if epoch % 1000 == 0:
      print("Epoch: ", epoch, " Best val error: ", best_validation_error)
      sys.stdout.flush()

  end_time = timeit.default_timer()
  time_required = (end_time - start_time) / 60.
  print((('Optimization complete. Best validation score of %f '
         'obtained at epoch %i, with test performance %f ') %
        (best_validation_error, best_epoch + 1, test_score)))
  print(('The code for file ' +
                        os.path.split(__file__)[1] +
                        ' ran for %.2fm' % time_required), file=sys.stderr)

  if verbose:
    np.set_printoptions(precision=4, suppress=True)
    print((classifier.get_params()))
  return {'train_losses': np.asarray(train_errors),
          'val_errors': np.asarray(validation_errors),
          'test_errors': np.asarray(test_errors),
          'classifier': classifier,
          'test_score': test_score,
          'val_score': this_validation_error,
          'best_val_score': best_validation_error,
          'best_epoch': best_epoch,
          'best_state': best_state,
          'num_active': classifier.get_active_units(),
          'runtime': time_required
          }


def usage():
  print((sys.argv[0] + "[-i id -d dataset -p extrapolationdataset -l layers -e epochs -n nodes" +
        " -r learningrate --l1=l1reg --l2=l2reg --shortcut --resfolder" +
        "  --gradient=sgd|adam --initfile=statefile -v -o]"))


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
  L1_reg = 0.00001
  L2_reg = 0.00001
  learning_rate = 0.01
  reg_start = 0
  reg_end = None
  output = False
  verbose = 0
  id = np.random.randint(0, 1000000)
  result_folder = "./"

  try:
    opts, args = getopt.getopt(sys.argv[1:], "hv:i:d:p:l:e:n:f:co",
                               ["help", "verbose=", "id=", "dataset=", "extrapol=", "layers=", "epochs=",
                                "nodes=", "l1=", "l2=", "lr=", "resfolder=",
                                "batchsize=", "initfile=", "gradient=",
                                "reg_start=", "reg_end=", "output"
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
      gradient = arg
    elif opt in ("--reg_start"):
      reg_start = int(arg)
    elif opt in ("--reg_end"):
      reg_end = int(arg)
    elif opt in ("-o", "--output"):
      output = True
    elif opt in ("-f", "--resfolder"):
      result_folder = arg

  # load dataset
  if not dataset_file:
    print("provide datasetfile!")
    usage()
    exit(1)
  dataset = utils.load_data(dataset_file)

  # load extrapolation test
  if len(extra_pol_test_sets) > 0:
    if verbose > 0:
      print("do also extrapolation test(s)!")
    extra_pols = [utils.load_data(test_set) for test_set in extra_pol_test_sets]

  if init_file:
    with open(init_file, 'rb') as f:
      init_state = pickle.load(f)
      print("load initial state from file " + init_file)

  if not os.path.exists(result_folder):
    os.makedirs(result_folder)

  name = result_folder + str(id)

  result = test_mlp(datasets=dataset, n_epochs=n_epochs, verbose=verbose > 0, learning_rate=learning_rate,
                    L1_reg=L1_reg, L2_reg=L2_reg, n_layer=n_layers, n_units=n_nodes, id=id,
                    gradient=gradient, batch_size=batch_size, init_state=init_state,
                    reg_start=reg_start, reg_end=reg_end
                    )

  classifier = result['classifier']
  with  open(name + '.best_state', 'wb') as f:
    pickle.dump(result['best_state'], f, protocol=pickle.HIGHEST_PROTOCOL)
  with  open(name + '.last_state', 'wb') as f:
    pickle.dump(classifier.get_state(), f, protocol=pickle.HIGHEST_PROTOCOL)

  extra_scores = []
  extra_scores_best = []
  for extra in extra_pols:
    extra_set_x, extra_set_y = extra[0]
    extra_scores.append(classifier.test_model(input=extra_set_x, labels=extra_set_y))
  # also for best_state
  classifier.set_state(result['best_state'])
  for extra in extra_pols:
    extra_set_x, extra_set_y = extra[0]
    extra_scores_best.append(classifier.test_model(input=extra_set_x, labels=extra_set_y))

  result_line = ""
  with  open(name + '.res', 'w') as f:
    if (id == 0):
      f.write('#C layers epochs nodes lr L1 L2 batchsize regstart regend' +
              ' id dataset gradient numactive bestepoch runtime' +
              "".join([' extrapol' + str(i) for i in range(1, len(extra_scores) + 1)]) +
              "".join([' extrapolbest' + str(i) for i in range(1, len(extra_scores_best) + 1)]) +
              ' valerror valerrorbest testerror\n')
      f.write('# extra datasets: ' + " ".join(extra_pol_test_sets) + '\n')
    result_line = [str(n_layers), str(n_epochs), str(n_nodes), str(learning_rate), str(L1_reg), str(L2_reg),
                   str(batch_size), str(reg_start), str(reg_end),
                   str(id), dataset_file, gradient,
                   str(result['num_active']), str(result['best_epoch']),
                   str(result['runtime'])] + \
                  [str(e) for e in extra_scores] + \
                  [str(e) for e in extra_scores_best] + \
                  [str(result['val_score']), str(result['best_val_score']), str(result['test_score'])]
    f.write(str.join('\t', result_line) + '\n')

  with open(name + '.validerror', 'wb') as csvfile:
    a = csv.writer(csvfile, delimiter='\t')
    a.writerows([["#C epoch", "val_error"]])
    a.writerows([["# "] + result_line])
    a.writerows(result['val_errors'])
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
