""" All essential utilities related to data and setup

AVAILABLE DATASETS :
  Iris

Module components
=================



# Data processing
#----------------

"""

import os
import sys
import code
import argparse
from functools import wraps

import numpy as np

from layers import PARAMETRIC_FUNCTIONS
from functions import ACTIVATIONS, OBJECTIVES
import network
from optimizers import get_optimizer

#==== ugly relative pathing hack to dataset
fpath = os.path.abspath(os.path.dirname(__file__))
path_to_dataset = fpath.rstrip(fpath.split('/')[-1]) + 'data'
#if not os.path.exists(path_to_dataset):
#    print('ERROR: Unable to locate project data directory')
#    print('Please restore the data directory to its original path at {}\n'.format(path_to_dataset),
#          'or symlink it to {}\n'.format(fpath),
#          'or specify the updated absolute path to the sandbox submodule scripts')
#    sys.exit()

if path_to_dataset not in sys.path:
    sys.path.append(path_to_dataset)
import dataset
from dataset import IrisDataset


#==============================================================================
#------------------------------------------------------------------------------
#                        Task-invariant utils
#------------------------------------------------------------------------------
#==============================================================================

#------------------------------------------------------------------------------
# Errors
#------------------------------------------------------------------------------
def verify_path(data_path, utils_path):
    """ """
    if not os.path.exists(path):
        h  = 'ERROR: Unable to locate project data directory'
        e1 = ('Please restore the data directory to ',
             'its original path at {}\n'.format(data_path))
        e2 = 'or symlink it to {}\n'.format(utils_path),
        e3 = 'OR specify the updated absolute path to the sandbox'
        print(h)
        print('{}{}{}'.format(h, e1, e2, e3))
        sys.exit()
    
    elif path_to_dataset not in sys.path:
        sys.path.append(path_to_dataset)



#------------------------------------------------------------------------------
# Data objects
#------------------------------------------------------------------------------

class AttrDict(dict):
    """ simply a dict accessed/mutated by attribute instead of index
    WARNING: cannot be pickled like normal dict/object
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
# Decorators
# ========================================
def TODO(f):
    """ Serves as a convenient, clear flag for developers and insures
        wrapee func will not be called
    """
    @wraps(f)
    def not_finished(*args, **kwargs):
        print('\n  {} IS INCOMPLETE'.format(f.__name__))
    return not_finished

def NOTIMPLEMENTED(f):
    """ Like TODO, but for functions in a class
        raises error when wrappee is called
    """
    @wraps(f)
    def not_implemented(*args, **kwargs):
        func_class = args[0]
        f_class_name = func_class.get_class_name()
        f_name = f.__name__
        msg = '\n  Class: {}, function: {} has not been implemented!\n'
        print(msg.format(f_class_name, f_name))
        raise NotImplementedError
    return not_implemented


def INSPECT(f):
    """ interupts computation and opens interactive python shell
    for user to evaluate wrapee function input/output

    NB: simply placing the `code.interact` line directly in the code
        location you would like to evaluate/debug is often more
        effective
    """
    @wraps(f)
    def inspector(*args, **kwargs):
        print('\n Inspecting function: <{}>'.format(f.__name__))
        x = args
        y = kwargs
        z = f(*args, **kwargs)
        code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        return z
    return inspector


#==============================================================================
#------------------------------------------------------------------------------
#                               Settings
#------------------------------------------------------------------------------
#==============================================================================


# Pathing
#------------------------------------------------------------------------------
# Data pathing
# ========================================
DATA_DIR = '../data/'
#==== Iris dataset
IRIS_DIR = DATA_DIR + 'Iris/'
IRIS_DATASET_PATH = IRIS_DIR + 'iris.npy'
#---- Iris train/test files
IRIS_TRAIN = IRIS_DIR + 'iris_train.npy'
IRIS_TEST  = IRIS_DIR + 'iris_test.npy'




####################### Not yet supported #####################################
"""
# Model pathing
# ========================================
#('m', 'model_name', MODEL_NAME_BASE, 'name to which model params saved'),
#('n', 'name_suff',  '', 'label or tag suffixing model name'),
# Model directory paths
MODEL_NAME_BASE = '{}_{}' # {data_label}_{layer_type}
MODEL_SAVE_PATH_ROOT = './Models/{}/' # Models/model_name/...
MODEL_PARAMS_PATH  = MODEL_SAVE_PATH_ROOT + 'Parameters/'
MODEL_RESULTS_PATH = MODEL_SAVE_PATH_ROOT + 'Results/'
MODEL_CHECKPT_PATH = MODEL_PARAMS_PATH    + 'checkpoints/' # save checkpoints
PYFILE_SAVE_PATH   = MODEL_SAVE_PATH_ROOT + 'original_files/' # overwrites

# Model results write path
RESULTS_BASE_NAME = 'X_{}'
PREDICTION_FNAME  = RESULTS_BASE_NAME.format('pred')
GROUNDTRUTH_FNAME = RESULTS_BASE_NAME.format('true')

# Model loss statistics
LOSS_PLOT_FNAME  = RESULTS_BASE_NAME.format('error') # visualization
LOSS_TRAIN_FNAME = RESULTS_BASE_NAME.format('train_error')
LOSS_TEST_FNAME  = RESULTS_BASE_NAME.format('test_error')
"""

# Dataset features
#------------------------------------------------------------------------------
# Iris dataset
# ========================================
IRIS = {'label' : 'iris',
        'path'  : IRIS_DATASET_PATH,
        'num_samples' : 150,
        'features_per_sample' : 4,
        'feature_split_idx' : 4,
        'classes' : {0 : 'Iris-setosa',
                     1 : 'Iris-versicolor',
                     2 : 'Iris-virginica'},
        }

# Parameters
#------------------------------------------------------------------------------
# Random seeds
RNG_SEED_DATA   = 9959  # for shuffling data into train/test evenly
RNG_SEED_PARAMS = 12345 # seeding parameter inits

# Hyperparameters
LEARNING_RATE = 0.01 # unused, opt defaults are good
CHANNELS = [IRIS['features_per_sample'], 163, len(IRIS['classes'])]



# Model/session configuration
#------------------------------------------------------------------------------

DEFAULT_CONFIGURATION = [
# ==== Data variables
('p', 'data_path',  DATA_DIR,  'relative path to dataset file'),
('s', 'seed',       RNG_SEED_PARAMS, 'int used for seeding random state'),
# ==== Model variables
('a', 'activation', 'sigmoid', '(lower-cased) activation func name'),
('d', 'dropout',    False, 'Whether to use dropout'),
('o', 'optimizer',  'sgd', '(lower-cased) optimizer name'),
('j', 'objective',  'softmax_cross_entropy', '(lower-cased) loss func name'),
('c', 'channels',   CHANNELS, 'list(int) layer sizes; more channels-->deeper'),
('l', 'learn_rate', LEARNING_RATE, 'optimizer learning rate'),
# ==== Training/session variables
('i', 'num_iters',  1500, 'number of training iterations'),
('b', 'batch_size', 6, 'training batch size: how many samples per iter'),
('v', 'verbose',    False, 'print model error while actively training'),
('_', 'dummy', False, 'dummy var workaround for jupyter notebook error'),
]


#==============================================================================
#------------------------------------------------------------------------------
#                                Data
#------------------------------------------------------------------------------
#==============================================================================

# One-hot encoding
#------------------------------------------------------------------------------
def to_one_hot(Y, num_classes=len(IRIS['classes'])):
    """ make one-hot encoding for truth labels

    Encodes a 1D vector of integer class labels into
    a sparse binary 2D array where every sample has
    length num_classes, with a 1 at the index of its
    constituent label and zeros elsewhere.

    Example
    -------
    Y = [3, 1, 1, 0, 2, 3, 2, 2, 2, 1]
    Y.shape = (10,)
    num_classes = 4
    one_hot shape == (10, 4)

    to_one_hot(Y) = [[0, 0, 0, 1], # 3
                     [0, 1, 0, 0], # 1
                     [0, 1, 0, 0], # 1
                     [1, 0, 0, 0], # 0
                     [0, 0, 1, 0], # 2
                     [0, 0, 0, 1], # 3
                     [0, 1, 0, 0], # 2
                     [0, 1, 0, 0], # 2
                     [0, 1, 0, 0], # 2
                     [1, 0, 0, 0], # 1
                    ]
    """
    # Dimensional integrity check on Y
    #  handles both ndim = 0 and ndim > 1 cases
    if Y.ndim != 1:
        Y = np.squeeze(Y)
        assert Y.ndim == 1

    # dims for one-hot
    n = Y.shape[0]
    d = num_classes

    # make one-hot
    one_hot = np.zeros((n, d))
    one_hot[np.arange(n), Y] = 1
    return one_hot.astype(np.int32)

# Loading dataset from disk
#------------------------------------------------------------------------------
def load_dataset(path):
    """ Loads dataset located at path """
    dataset = np.load(path)
    return dataset



#==============================================================================
#------------------------------------------------------------------------------
#                              Parser
#------------------------------------------------------------------------------
#==============================================================================

class Parser:
    """ Wrapper for argparse parser

    In addition to parsing STDIN args, Parser is a significant
    part of setup and configuration for training, as it specifies the
    majority of settings to use (through `default`)
    """
    P = argparse.ArgumentParser()
    def __init__(self):
        self.add_args(DEFAULT_CONFIGURATION)

    def add_args(self, defaults_config):
        """ formats default config to argparse readable form
        Argparse has a well defined standard library that has a
          lot of powerful utilities, but it is rather complex
          and there are certain constraints on the
          formatting of arguments added to the parser
        Example
        -------
        IN:  ('o', 'optimizer',  'sgd', '(lower-cased) optimizer name')

        OUT: ('-o', '--optimizer', type=str, default='sgd',
                    help='(lower-cased) optimizer name')

        The two current edge cases for parser args:
          - bools : require action='store_true' flag
          - lists : require nargs='+' flag
        """
        for k, name, d, h in defaults_config:
            #==== args: name
            args = ('-{}'.format(k), '--{}'.format(name))
            #==== kwargs: type, default, help, **
            if isinstance(d, bool):
                kw = {'action': 'store_true'}
            elif isinstance(d, list): # only have int lists
                kw = {'type':int, 'default':d, 'nargs':'+'}
            else:
                dtype = type(d)
                kw = {'type': dtype, 'default':d}
            kwargs = {**kw, **{'help': h}}
            #==== Add args to parser
            self.P.add_argument(*args, **kwargs)

    def parse_args_from_jupyter_notebook(self, inputs=None):
        """ workaround for parse_args call in notebook, ipython fine though"""
        inputs = '-_' if inputs is None else inputs
        return self.parse_args(inputs)

    def parse_args(self, inputs=None):
        """ parse_args, by default, reads from STDIN
        but can be directly called with the same format args
            eg: parser.parse_args('-i 3000 -a swish -b 10'.split(' '))
            or  parser.parse_args(['-i', '3000', '-a', 'swish, '-b', '10'])
        """
        if inputs is not None:
            ipt = inputs.split(' ') if isinstance(inputs, str) else inputs
            parsed = AttrDict(vars(self.P.parse_args(ipt)))
        else:
            parsed = AttrDict(vars(self.P.parse_args()))
        self.args = self.interpret_args(parsed)
        return parsed

    def interpret_args(self, parsed):
        # Remove jupyter-hack dummy var
        #----------------------------
        del parsed.dummy

        # Integrity input/output channels
        #----------------------------
        # input/output MUST adhere to dataset dims
        parsed.channels[0]  = CHANNELS[0]
        parsed.channels[-1] = CHANNELS[-1]

        # Activation
        #----------------------------
        pact = parsed.activation
        #==== Check if parameterized activation (Swish)
        activation = PARAMETRIC_FUNCTIONS.get(pact, None)
        if activation is None:
            #==== Check if valid activation Function
            activation = ACTIVATIONS.get(pact, None)
            if activation is None:
                # if None again, parsed activation arg is undefined in domain
                raise NotImplementedError('{} is undefined'.format(pact))
        # assign proper activation class
        parsed.activation = activation

        # Optimizer
        #----------------------------
        popt = parsed.optimizer
        opt = get_optimizer(popt) # raises ValueError if not defined
        parsed.optimizer = opt

        # Objective
        #----------------------------
        pobj = parsed.objective
        objective = OBJECTIVES.get(pobj, None)
        if objective is None: # then objective not in functions
            raise NotImplementedError('{} is undefined'.format(pobj))
        parsed.objective = objective
        return parsed


    def print_args(self):
        print('SESSION CONFIG\n{}'.format('='*79))
        margin = len(max(self.args, key=len)) + 1
        for k,v in self.args.items():
            print('{:>{margin}}: {}'.format(k,v, margin=margin))
        print('\n')


#==============================================================================
#------------------------------------------------------------------------------
#                      Training stats and other info
#------------------------------------------------------------------------------
#==============================================================================

# Status and results
#------------------------------------------------------------------------------
class SessionStatus: # TODO: really need to clean this up / split
    """ Provides information to user on current status and results of model

    In addition to helpful status updates, SessionStatus maintains
    the loss history collections for training or validation.
    """
    div1 = '#' + ('=' * 78)
    div2 = '#' + ('-' * 30)
    def __init__(self, model, opt, obj, num_iters, num_test):
        self.model_name = str(model)
        self.opt_name = str(opt)
        self.obj_name = str(obj)
        self.num_iters = num_iters
        self.num_test = num_test
        self.init_loss_history()
        self._get_network_arch(model)
        self.status_call_count = 0

    def init_loss_history(self):
        """ Initialize loss history containers for training and testing """
        # Each entry of form: (error, accuracy)
        self.train_history = np.zeros((self.num_iters, 2)).astype(np.float32)
        self.test_history  = np.zeros((self.num_test, 2)).astype(np.float32)

    def _get_network_arch(self, model):
        """ Formatted string of network architecture """
        # strings
        #-----------------
        arch = '{}\n  Layers: \n'.format(str(model)) # 'NeuralNetwork'
        line_layer    = '    {:>2} : {:<5} {}\n' # ParametricLayer, eg 'Dense'
        line_function = '          : {}\n'       # Function, eg 'Tanh'

        # Traverse layers
        #-----------------
        for unit in model.layers:
            unit_name   = str(unit)
            unit_class  = unit.__class__
            unit_module = unit.__module__

            #==== Parametric unit
            if unit_module == 'layers':
                kdims = unit.kdims
                layer_name, layer_num = unit_name.split('-')
                #==== Function case
                if unit_class in PARAMETRIC_FUNCTIONS.values():
                    # non-connection functions are not considered discrete
                    # layers within the network like Dense
                    layer_num = ' '
                arch += line_layer.format(layer_num, layer_name, kdims)
            #==== function.Function
            else:
                arch += line_function.format(unit_name)
        self.network_arch = arch


    def print_results(self, train=True, t=None):
        d2 = self.div2
        num_tr = self.num_iters
        num_test = self.num_test
        header = '\n# {} results, {} {}\n' + d2
        # Format header based on training or test
        if train:
            header = header.format('Training', num_tr, 'iterations')
            cut = int(num_tr * .2)
            loss_hist = self.train_history[cut:]
        else:
            header = header.format('Test', num_test, 'samples')
            loss_hist = self.test_history

        # Get stats on loss hist
        avg = np.mean(  np.copy(loss_hist), axis=0)
        q50 = np.median(np.copy(loss_hist), axis=0)

        # Format print lines
        line_key = '            Error   |  Accuracy'
        line_avg = '* Average: {:.5f}  |  {:.5f}'.format(avg[0], avg[1])
        line_q50 = '*  Median: {:.5f}  |  {:.5f}'.format(q50[0], q50[1])

        # Print results
        print(header)
        if t is not None:
            print('Elapsed time: {}'.format(t))
        print(line_key)
        print(line_avg)
        print(line_q50)
        print(d2)

    def summarize_model(self, train_results=False, test_results=False):
        arch = self.network_arch
        opt = self.opt_name
        obj = self.obj_name
        d1  = self.div1 #=====
        d2  = self.div2 #-----
        print(d1)
        print('# Model Summary: \n')
        print(arch)
        print('- OPTIMIZER : {}'.format(opt))
        print('- OBJECTIVE : {}'.format(obj))
        if train_results:
            self.print_results(train=True)
        if test_results:
            self.print_results(train=False)
        print(d1)

    def get_loss_history(self):
        return self.train_history, self.test_history

    def print_status(self, step, err, acc):
        title = '{:<5}:  {:^7}   |  {:^7}'.format('STEP', 'ERROR', 'ACCURACY')
        body  = '{:>5}:  {:.5f}   |   {:.4f}'
        i = step + 1
        e, a = float(err), float(acc)
        status = body.format(i, e, a)
        #==== Only print title on first call
        if self.status_call_count == 0:
            print('\n\n{}'.format(title))
        print(status)
        self.status_call_count += 1

    def __call__(self, step, err, acc, show=True, freq=100, test=False):
        loss_hist = self.test_history if test else self.train_history
        loss_hist[step] = err, acc
        if show and (test or ((step+1) % freq == 0)):
            if test and step == 0:
                self.status_call_count = 0
            self.print_status(step, err, acc)


#==============================================================================
# Classification functions
#==============================================================================

# Classification eval
#------------------------------------------------------------------------------
def get_predictions(Y_hat):
    """ Select the highest valued class labels in prediction from
    network output distribution
    We can approximate a single label prediction from a distribution
    of prediction values over the different classes by selecting
    the largest value (value the model is most confident in)

    Params
    ------
    Y_hat : ndarray.float32, (N, D)
        network output, "predictions" or scores on the D classes
    Returns
    -------
    Y_pred : ndarray.int32, (N,)
        the maximal class score, by index, per sample
    """
    Y_pred = np.argmax(Y_hat, axis=-1)
    return Y_pred

def indifferent_scores(scores):
    """ Check whether class scores are all close to eachother

    If class scores are all nearly same, it means model
    is indifferent to class and makes equally distributed
    values on each.

    This means that, even in the case where the model was
    unable to learn, it would still get 1/3 accuracy by default

    This function attempts preserve integrity of predictions
    """
    N, D = scores.shape
    if N > 1:
        mu = np.copy(scores).mean(axis=1, keepdims=True)
        mu = np.broadcast_to(mu, scores.shape)
    else:
        mu = np.full(scores.shape, np.copy(scores).mean())
    return np.allclose(scores, mu, rtol=1e-2, atol=1e-2)



def classification_accuracy(Y_hat, Y_truth, strict=False):
    """ Computes classification accuracy over different classes

    Params
    ------
    Y_pred : ndarray.float32, (N, D)
        raw class "scores" from network output for the D classes
    Y_truth : ndarray.int32, (N,)
        ground truth

    Returns
    -------
    accuracy : float
        the averaged percentage of matching predictions between
        Y_hat and Y_truth
    """
    if indifferent_scores(Y_hat):
        return 0.0

    if not strict:
        # Reduce Y_hat to highest scores
        Y_pred = get_predictions(Y_hat)

        # Take average match
        accuracy = np.mean(Y_pred == Y_truth)

    else:
        # Show raw class score
        Y = to_one_hot(Y_truth)
        scores = np.amax(Y * Y_hat, axis=1)
        accuracy = np.mean(scores)

    return accuracy


#==============================================================================
#------------------------------------------------------------------------------
#                               Trainer
#------------------------------------------------------------------------------
#==============================================================================


class Trainer:
    """ Manages the training for a model

    Trainer accepts a model, along with an optimizer
    and objective function, and trains it for the
    specified number of steps using a DataSet. # X=None, split_size=.8, seed=RNG_SEED_DATA
    """
    def __init__(self, channels, opt, obj, activation,
                 dataset=None, steps=1000, batch_size=6, dropout=False,
                 verbose=False, rng_seed=RNG_SEED_PARAMS):
        self.channels = channels
        self.opt = opt()
        self.obj = obj()
        self.activation = activation
        self.steps = steps
        self.batch_size = batch_size
        self.verbose = verbose
        self.rng_seed = rng_seed
        self.init_network(activation, dropout)
        self.init_dataset(dataset)
        self.init_session_status()

    def init_network(self, act, use_dropout=False):
        """ seed and instantiate network """
        np.random.seed(self.rng_seed)
        self.model = network.NeuralNetwork(self.channels, act, use_dropout)

    def init_dataset(self, dataset):
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = IrisDataset()
        self.num_test = self.dataset.X_test.shape[0]

    def init_session_status(self):
        model = self.model
        opt = self.opt
        obj = self.obj
        steps = self.steps
        #num_test = self.dataset.X_test.shape[0]
        num_test = self.num_test
        self.session_status = SessionStatus(model, opt, obj, steps, num_test)

    def summarize_results(self):
        seed = self.rng_seed
        d2 = '-'*60
        num_tr = self.steps
        num_test = self.num_test
        header = '\n# Model: {}\n#  Seed: {}\n' + d2
        header = header.format(self.channels, self.rng_seed)

        # Get stats on train history
        #  skip first 100 iters
        avg = np.mean(  np.copy(self.train_history[100:]), axis=0)
        q50 = np.median(np.copy(self.train_history[100:]), axis=0)

        tavg = np.mean(np.copy(self.test_history), axis=0)
        tq50 = np.median(np.copy(self.test_history), axis=0)

        # Print results
        print(header)
        print('   TRAIN    Error  |  Accuracy')
        print('* Average: {:.5f} | {:.5f}'.format(avg[0], avg[1]))
        print('*  Median: {:.5f} | {:.5f}'.format(q50[0], q50[1]))
        print(d2)
        print('    TEST    Error  |  Accuracy')
        print('* Average: {:.5f} | {:.5f}'.format(tavg[0], tavg[1]))
        print('*  Median: {:.5f} | {:.5f}'.format(tq50[0], tq50[1]))
        print(d2)
        print('\n\n')

    def get_loss_histories(self):
        lhtr, lhte =  self.train_history, self.test_history
        return lhtr, lhte


    def train(self):
        v = self.verbose
        for step in range(self.steps):
            # batch data
            #------------------
            x, y = self.dataset.get_batch(step, self.batch_size)

            # forward pass
            #------------------
            y_hat = self.model.forward(x)
            error, class_scores = self.obj(y_hat, y)
            accuracy = classification_accuracy(class_scores, y)
            self.session_status(step, error, accuracy, show=v)

            # backprop and update
            #------------------
            grad_loss = self.obj(backprop=True)
            self.model.backward(grad_loss)
            self.model.update(self.opt)

    def evaluate(self):
        num_steps = self.dataset.X_test.shape[0]
        v = self.verbose
        for i in range(num_steps):
            x, y = self.dataset.get_batch(i, test=True)

            # forward pass
            #------------------
            y_hat = self.model.forward(x, test=True)
            error, class_scores = self.obj(y_hat, y)
            accuracy = classification_accuracy(class_scores, y)
            self.session_status(i, error, accuracy, show=v, test=True)

    def __call__(self):
        self.train()
        self.evaluate()
        self.train_history, self.test_history = self.session_status.get_loss_history()
        self.summarize_results()
