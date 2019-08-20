"""
SVR from sklearn

"""
import time
import sys
import timeit
import getopt

import numpy
import pickle

from sklearn.svm import SVR

from .utils import *



__docformat__ = 'restructedtext en'

def evaluate_svr(x,model):
  predictions = []
  for (d, svr) in model:
    predictions.append(svr.predict(x))

  return np.transpose(np.asarray(predictions))

def test_svr(x,y, model):
  errors = []
  for (d, svr) in model:
    pred_y = svr.predict(x)
    errors.append(np.mean(np.square(y[:, d] - pred_y)))

  return np.mean(errors)

def train_test_svr(datasets, C=1.0, epsilon=0.001, gamma=0.1,
                   model=None,
                   id=None,
                   init_file=None):
  """
  :type datasets: ((matrix,matrix),(matrix,matrix),(matrix,matrix))
  :param datasets: ((train-x,train-y),(valid-x,valid-y),(test-x,test-y))

 """

  train_set_x, train_set_y = datasets[0]
  valid_set_x, valid_set_y = datasets[1]
  if len(datasets) > 2:
    test_set_x, test_set_y = datasets[2]
  else:
    test_set_x = test_set_y = None

  inputdim = len(datasets[0][0][0])
  outputdim = len(datasets[0][1][0])
  print("Input/output dim:", (inputdim, outputdim))
  print("Training set, val set:", (train_set_x.shape[0], valid_set_x.shape[0]))

  start_time = timeit.default_timer()

  # need one SVR for each output dimension
  rng = numpy.random.RandomState(int(time.time()) if id is None else id)
  if model is None:
    model = [(d, SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)) for d in range(outputdim)]
  if init_file:
    model = pickle.loads(init_file)

  ###############
  # TRAIN MODEL #
  ###############
  print('... training')
  sys.stdout.flush()

  for (d, svr) in model:
    svr.fit(train_set_x, train_set_y[:,d])

  validation_error = test_svr(valid_set_x, valid_set_y, model)
  if test_set_x is not None:
    test_error       = test_svr(test_set_x, test_set_y, model)
  else:
    test_error = np.inf

  end_time = timeit.default_timer()
  time_required = (end_time - start_time) / 60.
  print((('Optimization complete. Best validation score of %f and test performance %f ') %
        (validation_error, test_error)))

  return {'classifier': model,
          'test_score': test_error,
          'val_score':  validation_error,
          'runtime': time_required
          }

def usage():
  print((sys.argv[0] + "[-i id -d dataset -p extrapolationdataset -C costfactor -e epsilon -g gamma" +
                      " --resfolder -v"))

if __name__ == "__main__":
  dataset_file = None
  extra_pol_test_sets = []
  extra_pols = []
  init_file = None
  epsilon = 0.1
  Ccost=1.0
  gamma = 1.0
  verbose = 0
  id = np.random.randint(0, 1000000)
  result_folder = "./"
  num_points=None

  try:
    opts, args = getopt.getopt(sys.argv[1:], "hv:i:d:p:e:g:f:C:o",
                               ["help", "verbose=", "id=", "dataset=", "extrapol=",
                                "cost=", "epsilon=",
                                "gamma=", "resfolder=",
                                "initfile=", "num_points="
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
    elif opt in ("-C", "--cost"):
      Ccost = float(arg)
    elif opt in ("-e","--epsilon"):
      epsilon = float(arg)
    elif opt in ("-g","--gamma"):
      gamma = float(arg)
    elif opt in ("--initfile"):
      init_file = arg
    elif opt in ("-o", "--output"):
      output = True
    elif opt in ("-f", "--resfolder"):
      result_folder = arg
    elif opt in ("--num_points"):
      num_points = int(arg)


  # load dataset
  if not dataset_file:
    print("provide datasetfile!")
    usage()
    exit(1)
  dataset = load_data(dataset_file)

  # restrict
  num_pts = len(dataset[0][0])
  if num_points is None:
    num_points = num_pts
  if num_points > num_pts:
    num_points = num_pts
  if num_points != num_pts:
    print("retrict dataset to use: " + str(num_points))
    datasetnew = ((dataset[0][0][:num_points, :], dataset[0][1][:num_points, :]), dataset[1])
    if len(dataset) > 2:
      dataset = datasetnew + (dataset[2],)
    else:
      dataset = datasetnew

  # load extrapolation test
  if len(extra_pol_test_sets) > 0:
    if verbose > 0:
      print("do also extrapolation test(s)!")
    extra_pols = [load_data(test_set) for test_set in extra_pol_test_sets]

  if not os.path.exists(result_folder):
    try:
      os.makedirs(result_folder)
    except OSError:
      pass

  name = result_folder + str(id)

  result = train_test_svr(datasets=dataset, C=Ccost, epsilon=epsilon, gamma=gamma, id=id,  init_file=init_file)

  classifier = result['classifier']
  with  open(name + '.last_state', 'wb') as f:
    pickle.dump(classifier, f)

  extra_scores = []
  for extra in extra_pols:
    extra_set_x, extra_set_y = cast_dataset_to_floatX(extra[0])
    extra_scores.append(test_svr(extra_set_x, extra_set_y, classifier))

  result_line = ""
  with  open(name + '.res', 'w') as f:
    if (id == 0):
      f.write('#C Ccost epsilon gamma' +
              ' id dataset' +
              ' runtime' +
              "".join([' extrapol' + str(i) for i in range(1, len(extra_scores) + 1)]) +
              ' valerror testerror\n')
      f.write('# extra datasets: ' + " ".join(extra_pol_test_sets) + '\n')
    result_line = [str(Ccost), str(epsilon), str(gamma),
                   str(id), dataset_file, str(result['runtime'])] + \
                  [str(e) for e in extra_scores] + \
                  [str(result['val_score']), str(result['test_score'])]
    f.write(str.join('\t', result_line) + '\n')
