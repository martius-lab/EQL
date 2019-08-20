"""
 Utility functions

"""
import csv
import numpy as np
import theano
from itertools import chain
import os
import gzip
import pickle
#import dill


__docformat__ = 'restructedtext en'


def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def relative_prob(x):
    e_x = (x - np.min(x))
    out = e_x / e_x.sum()
    return out

def sample_from_dist(pdf,rnd=None):
    return samples_from_dist(pdf, 1, rnd)[0]


def samples_from_dist(pdf,n=1,rnd=None):
    if rnd is None:
        return np.random.choice(len(pdf),n,p=pdf)
    else:
        return rnd.choice(len(pdf),n,p=pdf)

def samples_distinct_from_dist(pdf,n=1,rnd=None):
    samples=list(set(samples_from_dist(pdf,3*n,rnd)))
    if len(samples)<n:
        samples=list(set(samples_from_dist(pdf,50*n,rnd)))
    if len(samples)<n:
        return samples
    else:
        return samples[:n]

def is_sequence(obj):
    return hasattr(obj, '__len__') and hasattr(obj, '__getitem__')


def flatten(l):
    return list(chain.from_iterable(l))


def normalize(vec):
    n = np.linalg.norm(vec)
    if n > 0:
        return vec / n
    else:
        return vec


def add_diagonal_limit(mat, val, max_size):
    di = np.diag_indices(min(min(mat.shape), max_size), mat.ndim)
    mat[di] += val

def wrap_mat_to_vec_func_3(func, *args):
    return lambda p1, p2, p3: func(np.expand_dims(p1, axis=0),
                                   np.expand_dims(p2, axis=0),
                                   np.expand_dims(p3, axis=0), *args)


def wrap_mat_to_vec_func_3_0(func, *args):
    return lambda p1, p2, p3: func(np.expand_dims(p1, axis=0),
                                   np.expand_dims(p2, axis=0),
                                   np.expand_dims(p3, axis=0), *args)[0]

def wrap_mat_to_vec_func_2(func, *args):
    return lambda p1, p2: func(np.expand_dims(p1, axis=0),
                               np.expand_dims(p2, axis=0),
                               *args)


def wrap_mat_to_vec_func_2_0(func, *args):
    return lambda p1, p2: func(np.expand_dims(p1, axis=0),
                               np.expand_dims(p2, axis=0),
                               *args)[0]


def cast_dataset_to_floatX(data_xy):
    data_x, data_y = data_xy
    return np.asarray(data_x,dtype=theano.config.floatX),np.asarray(data_y,dtype=theano.config.floatX)


def cast_to_floatX(array):
    return np.asarray(array, dtype=theano.config.floatX)


def load_from_hold(name):
    assert (not name is None)
    if os.path.exists(name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            print("loaded data from the file " + name)
    else:
        print("Initialising with an empty list")
        data = []
    return data


def dump_for_hold(data, name):
    print("Dumping:", name)
    assert (not name is None) and (not data is None)
    with  open(name , 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path):
            dataset = new_path

    if not os.path.isfile(dataset):
        print('cannot find dataset', dataset)
        return
    print('... loading data ' + dataset)
    # Load the dataset
    f = gzip.open(dataset, 'rb')
#    try:
    datasets = pickle.load(f, encoding='latin1')
#    except ImportError:
#        f.close()
#        f = gzip.open(dataset, 'rb')
#        datasets = dill.load(f)
    f.close()

    return datasets

def splitDataSet(inputs, outputs):
    assert len(inputs) == len(outputs)
    size = len(inputs)
    ts = size * 80 / 100
    vs = size * 10 / 100
    train_set = (inputs[:ts], outputs[:ts])
    valid_set = (inputs[ts:ts + vs], outputs[ts:ts + vs])
    test_set = (inputs[ts + vs:], outputs[ts + vs:])
    return train_set, valid_set, test_set


def splitDataSetShuffle(inputs, outputs,percent_val_test=10):
    assert len(inputs) == len(outputs)
    size = len(inputs)
    shuffle = np.random.permutation(size)
    inps = np.asarray(inputs)[shuffle]
    outs = np.asarray(outputs)[shuffle]
    ts = size * (100-2*percent_val_test) / 100
    vs = size * percent_val_test / 100
    train_set = (inps[:ts], outs[:ts])
    valid_set = (inps[ts:ts + vs], outs[ts:ts + vs])
    test_set = (inps[ts + vs:], outs[ts + vs:])
    return train_set, valid_set, test_set

def splitDataSetNoTest(inputs,outputs):
    assert len(inputs) == len(outputs)
    size=len(inputs)
    ts=size*90/100
    train_set=(inputs[:ts],outputs[:ts])
    valid_set=(inputs[ts:],outputs[ts:])
    return train_set, valid_set

def addTestSet(train_val,test_set):
    return train_val[0], train_val[1], test_set

# cuts dataset into those where the input vectors that have a maxnorm smaller or equal to cut and the rest
def cutDataSet(inputs,outputs, cut):
    sel = np.linalg.norm(inputs,ord=np.inf,axis=1) <= cut
    # sel = np.array([not(all(x<=cut) and all(x>=-cut)) for x in inputs])
    return (inputs[sel], outputs[sel]),(inputs[np.logical_not(sel)], outputs[np.logical_not(sel)])

def splitTrainValSets(inputs,outputs,cut):
    data_full=splitDataSetNoTest(inputs,outputs)
    (train_all,val_all) = data_full
    dat_sel = cutDataSet(train_all[0],train_all[1], cut)
    return data_full, dat_sel

def addNoise(data, size):
    noise = np.random.normal(0,size,data.shape)
    return data + noise

def loadState(filename):
    with open(filename, "rb") as f:
        return pickle.load(f, encoding='latin1')


def readCSVTable(filename, dtype='|S40'):
    data = []
    comments = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            if row[0][0] != '#':
                data.append(row)
            else:
                comments.append(row)
    return (np.asarray(data, dtype=dtype), comments)


def getIdx(header, colname): return np.asscalar(np.where(np.asarray(header) == colname)[0])
