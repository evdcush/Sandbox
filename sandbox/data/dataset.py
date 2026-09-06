""" Utilities for working with datasets

# Dataset acquisition
#--------------------
 - Small (file-size) datasets can be stored in this data directory,
   but larger datasets will need to be sourced by user
 - Some basic utilties (like `sub_wget_data`) can help acquire a
   dataset from source

# Dataset management
#-------------------
How data is read, processed, serialized, split, normalized, etc is
most likely going to be per-dataset, per-task process. Datasets have
the following functionality : 
  - loads dataset or a subset of a dataset
  - shuffles samples within dataset
  - split train/test sets (if not already split)

# Dataset class
#--------------
Given that the iris dataset is currently the only available dataset,
the IrisDataset is the most atomic or base dataset to work with.

When more datasets are added, if there is need for a standard interface
or too much boilerplate, a base class may be added.

"""
import os
import sys

import numpy

#------------------------------------------------------------------------------
#                               Constants
#------------------------------------------------------------------------------
# dir paths
# =========
PATH_DATA_DIR = str(os.path.abspath(os.path.dirname(__file__)))
PATH_IRIS_DIR = PATH_DATA_DIR + '/Iris/'

# file paths
# ==========
PATH_IRIS_FILE  = PATH_IRIS_DIR + 'iris.npy'
PATH_IRIS_TRAIN = PATH_IRIS_DIR + 'iris_train.npy'
PATH_IRIS_TEST  = PATH_IRIS_DIR + 'iris_test.npy'

# Seeds
# =====
RNG_SEED_IRIS = 9959  # for even class distribution in train/test


#==============================================================================
#------------------------------------------------------------------------------
#                               Datasets
#------------------------------------------------------------------------------
#==============================================================================

# Iris features
# =============
IRIS = {'label': 'iris',
        'path': PATH_IRIS_FILE,
        'num_samples': 150,
        'features_per_sample': 4,
        'feature_split_idx': 4,
        'num_classes': 3,
        'classes' : {0: 'Iris-setosa',
                     1: 'Iris-versicolor',
                     2: 'Iris-virginica'},
        }

#==============================================================================
#                               Functions
#==============================================================================

# Dataset load funcs
#------------------------------------------------------------------------------
def load_dataset(dpath): # assumes npy serialization
    assert os.path.exists(dpath) and dpath[-4:] == '.npy'
    return numpy.load(dpath)

def load_dataset_iris_full():
    return load_dataset(PATH_IRIS_FILE)

def load_dataset_iris_train_eval():
    return load_dataset(PATH_IRIS_TRAIN), load_dataset(PATH_IRIS_TEST)


# Data processing functions
#------------------------------------------------------------------------------

def get_batch(X, step, batch_size=1, test=False):
    """ Batches samples from dataset X
    ASSUMED: batch_size is a factor of the number of samples in X
    #==== Training
    - Batches are selected randomly *and without replacement* wrt
      the previous batches.
      This is done based on current step:
        When current step has become a multiple of the number
        of samples in X, the samples positions in X are
        randomly shuffled.
    #==== Testing
    - Batches are selected sequentially, with no randomness
    Params
    ------
    X : ndarray, (features & labels)
        X is an array containing both the features (x) and the labe
        or classes (y). split(X) ---> x, y
    step : int
        current training iteration
    batch_size : int
        number of samples in minibatch
    test : bool
        whether X is a test set. If test, no random ops.
    Returns
    -------
    x : ndarray, (batch_size, ...)
        batch data features
    y : ndarray.int32, (batch_size,)
        batch ground truth labels (or class)
    """
    assert batch_size > 0 and isinstance(batch_size, int)
    # Get dimensions and indices
    #-------------------------------
    N = X.shape[0]
    b = batch_size if batch_size <= N else batch_size % N
    #==== Batching indices
    i = (b * step) % N  # start index [inclusive]
    j = i + b           # end   index (exclusive)
    
    # Check if need for reshuffle
    #-------------------------------
    if i == 0 and not test: # test never shuffled
        numpy.random.shuffle(X)
    
    # Get batch and split
    #-------------------------------
    batch = numpy.copy(X[i:j])
    x, y = numpy.split(batch, [IRIS['feature_split_idx']], axis=1)
    y = y.astype(numpy.int32)
    #==== Squeeze Y to 1D
    y = numpy.squeeze(y) if b > 1 else y[:,0]
    return x, y

# Check equal distribution classes
#----------------------------------
def equally_distributed_classes(X):
    # Assumes X of shape (N,...,D), where D is class labels
    _, counts = numpy.unique(X[...,-1], return_counts=True)
    return numpy.all(counts == (X.shape[0] // IRIS['num_classes']))



# Split into Train/Test sets
#-----------------------------
def split_dataset(X, split_size=.8, split_seed=RNG_SEED_IRIS):
    """ Splits a dataset X into training and testing sets.
    Note: RNG_SEED_IRIS value was chosen because it evenly distribute
          the number of Iris classes between train and test sets,
          at split_size=.8
          (eg, for both train and test, the number of samples for
           num_setosa == num_versicolor == num_virginica)
    Params
    ------
    X : ndarray
        Primary dataset. split indices are instantiated
        on the first axis of X.shape, which is assumed to be
        the number of samples.
    split_size : float
        split percentage where N * split_size total samples are used
        for training, and the remaining for testing.
    split_seed : int
        used for seeding numpy's random module for consistent splits
        on X to produce training and testing sets
    Returns
    -------
    X_train, X_test : ndarray,
        train and test shapes, respectively are
        (N * split_size, D), (N * (1 - split_size), D)
        eg, for .8 split on 150-sample set:
            (120, D), (30, D)
    """
    # Get split vars
    #--------------------------
    split_size = split_size
    split_seed = split_seed
    assert 0.0 < split_size < 1.0

    # Spliting index
    #--------------------------
    N = X.shape[0]
    split_idx = [int(N * split_size)]

    # Seed, permute, split
    #--------------------------
    numpy.random.seed(split_seed)
    X_shuffled = numpy.random.permutation(X)
    X_train, X_test = numpy.split(X_shuffled, split_idx)
    return X_train, X_test



#==============================================================================
#                               Functions
#==============================================================================
# conveniently wraps the functions above

class IrisDataset:
    """ Maintains all info and utils related to the Iris dataset
    Attributes are simply extracted from IRIS
    """
    path = PATH_IRIS_DIR
    feature_split_idx = 4
    num_classes = 3
    classes = {0 : 'Iris-setosa', 1 : 'Iris-versicolor', 2 : 'Iris-virginica'}
    def __init__(self, split_size=.8, seed=RNG_SEED_IRIS, resplit=False):
        self.split_size = split_size
        self.split_seed = seed
        self.resplit = resplit
        self.load_dataset()

    # Load datasets
    #-----------------------------
    def load_dataset(self):
        if self.resplit:
            X = load_dataset_iris_full()
            x_train, x_test = self.split_dataset(X)
        else:
            # otherwise just load train/test sets
            x_train = load_dataset(PATH_IRIS_TRAIN)
            x_test  = load_dataset(PATH_IRIS_TEST)
        self.X_train = x_train
        self.X_test  = x_test

    # Data processing
    #---------------------------
    def split_dataset(self, X):
        Y = numpy.copy(X)
        tr, ev = split_dataset(Y, self.split_size, self.split_seed)

        # Check whether equal distrib classes possible with split
        if int(Y.shape[0] * self.split_size) % self.num_classes != 0:
            return tr, ev

        # Even distrib possible, split until satisfied
        cap = 100
        while not equally_distributed_classes(ev) and cap > 0:
            tr, ev = split_dataset(Y, self.split_size, None)
            cap -= 1
        return tr, ev

    def get_batch(self, step, batch_size=1, test=False):
        X = self.X_test if test else self.X_train
        return get_batch(X, step, batch_size, test=test)


