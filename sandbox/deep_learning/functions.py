""" All math functions and ops used by a model

This module provides the foundation of functions and operations
to build a network and optimize models.

It contains only the units that would used within a model.
Other functions, for training or data processing, can be found in `to_

Module components
=================
Function : base class for functions

atomic functions : elementary functions and factors
    Log, Square, Exp, Power, Sqrt

connection functions : functions using learnable variables
    Bias, Matmul, Linear

ReductionFunction : functions that reduce dimensionality
     Sum, Mean, Prod, XMax, XMin

activation functions : nonlinearities
    ReLU, ELU, SeLU, Softplus, Sigmoid, Tanh, Softmax, Swish

loss functions : objectives for gradient descent
    LogisticCrossEntropy, SoftmaxCrossEntropy

"""
import code
import numpy as np


#------------------------------------------------------------------------------
# Helpers and handy decorators
#------------------------------------------------------------------------------

class AttrDict(dict):
    """ dict accessed/mutated by attribute instead of index """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def to_one_hot(Y, num_classes=3):
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


#==============================================================================
#                              Function
#==============================================================================

# Base function class
#------------------------------------------------------------------------------

# Function
# --------
# inherits :
# derives  : ReductionFunction
class Function:
    """ Function parent class for various, mostly mathematical ops

    Function defines the basic structure and interface to most of
    the functions used in network ops

    In addition to ops and attributes described below, most functions
    inherit the following, unchanged, from Function:
     # __init__
     # __str__
     # __repr__
     # __call__
     # Function.cache

    Function ops
    ------------
    forward : f(X)
        the function
    backward : f'(X)
        the derivative of the function (wrt chained gradient)

    Attributes
    ----------
    name : str
        name of the function (based on function class name)
    cache : object
        cache is used for whatever data the functions store for
        retrieval during backpropagation (eg, inputs/outputs or
        reduction axes)
        WARNING: cache is automatically cleared after access
    test : bool
        whether function is being called in testing; prevents
        caching mostly
    """
    def __init__(self, *args, **kwargs):
        self.name = self.__class__.__name__
        self._cache = None
        self.test = False
        for attribute, value in kwargs.items():
            setattr(self, attribute, value)

    def __str__(self,):
        # str : $classname
        #     eg 'MatMul'
        return self.name

    def __repr__(self):
        # repr : functions.$classname
        #    eg "functions.MatMul"
        rep = '"functions.{}()"'.format(self.__class__.__name__)
        return rep

    @property
    def cache(self):
        """ NOTE: 'clears' cache upon access
        (since it is always reset in backprop)
        """
        if self._cache is not None:
            cache_objs = self._cache
            self._cache = None
            return cache_objs

    @cache.setter
    def cache(self, *fvars):
        if not self.test:
            self._cache = fvars if len(fvars) > 1 else fvars[0]

    def forward(self, X, *args):
        """ forward serves as an interface to a staticmethod
        for most Functions (ALL functions, currently).

        These methods are purely functional, and perform the function
        represented by the class.

        for the majority of functions, 'forward' simply caches
        the input, and calls it's class' staticmethod.

        Function, as the base class, provides a simple implementation
        of this behavior.

        If the Function class uses a staticmethod as such,
        the methods are generally the lower-case version of the class name,
        and the derivative functions have '_prime' concatenated.

        For example:
            Tanh : tanh, tanh_prime
            Sqrt : sqrt, sqrt_prime
            MatMul : matmul, matmul_prime

        Some Functions, however, have forward or backward processes
        that have more significant side-effects or other constraints
        that would not be complete by the pure functions alone, the
        most common being additional input arguments
          - LogisticCrossEntropy, for instance, cannot use its
            functions alone (and, incidentally, does not have
            conforming function names).

        While Function's base forward and backward functions cover
        the majority of functions' needs, I keep the boilerplate
        there anyway, because I think it makes things clearer,
        especially to those unfamiliar with chaining.

        """
        # in this case, self.name = 'Function'
        fn_name = self.name.lower()
        if hasattr(self, fn_name): # has a 'self.function' ?
            fn = getattr(self, fn_name)
            self.cache = X
            return fn(X, *args)
        else:
            raise NotImplementedError

    def backward(self, gY, *args):
        fn_name = self.name.lower() + '_prime'
        if hasattr(self, fn_name):
            fn = getattr(self, fn_name)
            X = self.cache
            gX = gY * fn(X, *args)
            return gX
        else:
            raise NotImplementedError

    def __call__(self, *args, backprop=False, test=False, **kwargs):
        self.test = test
        func = self.backward if backprop else self.forward
        return func(*args, **kwargs)


#==============================================================================
#                             Math Functions
#==============================================================================

#------------------------------------------------------------------------------
#  atomic functions
#------------------------------------------------------------------------------

class Exp(Function): ##
    """ Exponential function """
    @staticmethod
    def exp(x):
        return np.exp(x)

    @staticmethod
    def exp_prime(x):
        return np.exp(x)

    def forward(self, X):
        Y = self.exp(X)
        self.cache = Y
        return Y

    def backward(self, gY):
        Y = self.cache
        gX = gY * Y
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Log(Function): #
    """ Natural logarithm function """
    @staticmethod
    def log(x):
        return np.log(x)

    @staticmethod
    def log_prime(x):
        return 1.0 / x

    def forward(self, X):
        self.cache = X
        Y = self.log(X)
        return Y

    def backward(self, gY):
        X = self.cache
        gX = gY * self.log_prime(X)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Square(Function): #
    """ Square """
    @staticmethod
    def square(x):
        return np.square(x)

    @staticmethod
    def square_prime(x):
        return 2.0 * x

    def forward(self, X):
        self.cache = X
        Y = self.square(X)
        return Y

    def backward(self, gY):
        X = self.cache
        gX = gY * self.square_prime(X)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Sqrt(Function): # #
    """ Square root """
    @staticmethod
    def sqrt(x):
        return np.sqrt(x)

    @staticmethod
    def sqrt_prime(x):
        return 1.0 / (2 * np.sqrt(x))

    def forward(self, X):
        self.cache = X
        Y = self.sqrt(X)
        return Y

    def backward(self, gY):
        X = self.cache
        gX = gY * self.sqrt_prime(X)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Power(Function): #
    """ pow function """
    @staticmethod
    def power(x, p):
        return np.power(x, p)

    @staticmethod
    def power_prime(x, p):
        return p * np.power(x, (p-1))

    def forward(self, X, p):
        self.cache = X, p
        Y = self.power(X, p)
        return Y

    def backward(self, gY):
        X, p = self.cache
        gX = gY * self.power_prime(X, p)
        return gX



#------------------------------------------------------------------------------
# Connection functions :
#     Functions using learnable variables
#------------------------------------------------------------------------------

class Bias(Function): # #
    """ Adds to matrix X a bias vector B

    Params
    ------
    X : ndarray.float32, (N, D)
        input matrix data
    B : ndarray.float32, (D,)
        bias array
    """
    @staticmethod
    def bias(x, b):
        return x + b

    @staticmethod
    def bias_prime(x, b):
        """ derivative of bias is one """
        dx = np.ones_like(x, 'f')
        db = np.ones_like(b, 'f')
        return dx, db

    def forward(self, X, B):
        #self.cache = X
        Y = self.bias(X, B)
        return Y

    def backward(self, gY, B):
        #X = self.cache
        #dX, dB = self.bias_prime(X, B)
        #gX = gY * dX
        #gB = (gY * dB).sum(axis=0)
        return gY, gY.sum(0)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class MatMul(Function): # #
    """ Matrix multiplication function defined on 2D matrices

    Assumes: matrices X, Y are properly ordered as input args
        In other words, X.shape[-1] == Y.shape[0]

    Params
    ------
    X : ndarray
        2D matrix with shape (N, K), N an arbitrary int
    Y : ndarray
        2D matrix with shape (K, P), P an arbitrary int

    Returns
    -------
    Z : ndarray
        2D matrix product of XY, with shape (N, P)
    """
    @staticmethod
    def matmul(x, y):
        return np.matmul(x, y)

    @staticmethod
    def matmul_prime(x, y):
        dx = y.T
        dy = x.T
        return dx, dy

    def forward(self, X, Y):
        self.cache = np.copy(X)
        Z = self.matmul(X, Y)
        return Z

    def backward(self, gZ, Y):
        X = self.cache
        dX, dY = self.matmul_prime(X, Y)
        gX = self.matmul(gZ, dX) # (N, K).(K, D)
        gY = self.matmul(dY, gZ) # (D, N).(N, K)
        return gX, gY

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Linear(Function): ##
    """ Affine transformation function, Y = X.W + B
    See also: MatMul and Bias, which Linear composes

    Params
    ------
    X : ndarray.float32, (N, D)
        2D data matrix, N samples, D features
    W : ndarray.float32, (D, K)
        2D weights matrix mapped to features
    B : ndarray.float32, (K,)
        1D bias vector

    Returns
    -------
    Y : ndarray.float32, (N, K)
        transformation on data X
    """
    @staticmethod
    def linear(x, w, b):
        y = MatMul.matmul(x, w)
        z = Bias.bias(y, b)
        return z

    @staticmethod
    def linear_prime(x, w, b):
        dX, dW = MatMul.matmul_prime(x, w)
        _,  dB = Bias.bias_prime(x, b)
        return dX, dW, dB

    def forward(self, X, W, B):
        self.cache = np.copy(X)
        Z = self.linear(X, W, B)
        return Z

    def backward(self, gZ, W, B):
        X = self.cache
        dX, dW, dB = self.linear_prime(X, W, B)
        gX = MatMul.matmul(gZ, dX) # (N, K).(K, D)
        gW = MatMul.matmul(dW, gZ) # (D, N).(N, K)
        gB = np.sum(gZ * dB, axis=0)
        return gX, gW, gB


#==============================================================================
#------------------------------------------------------------------------------
#                            ReductionFunctions
#------------------------------------------------------------------------------
#==============================================================================

# ReductionFunction
# -----------------
# inherits : Function
# derives  : Sum, Mean, Prod, Max Min
class ReductionFunction(Function): # #
    """ Function class for op that reduce dimensionality

    Due to some extra complexity in broadcasting algebra wrt
    reduced or missing dimensions, reduction functions have an
    extra instance attribute for the reduction kwargs (axis, keepdims).
        This allows for easy dimensionality restoration during
        backprop
    """
    def __init__(self, *args, **kwargs):
        self.flags = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def tupleify(x):
        return (x,) if isinstance(x, int) else tuple(x)

    def format_axes(self, axes, shape):
        if axes is None: #==== same as applying along all axes
            axes = range(len(shape))
        return self.tupleify(axes)

    def restore_dims(self, Y):
        """ Restore any non-leading missing dims from Y
        There are a few constraints to NumPy broadcasting,
        which defines how operations involving arrays with
        different dimensions can still be applied.

        For automatic broadcasting, the arrays
        involved must satisfy one of the following
        3 properties:

        Properties
        ==========
        1 - exactly same shape
        >>----->  X.shape == Y.shape

        2 - Same num of dims, and len of each dim is either
            the same, or 1
        >>-----> X.ndim == Y.ndim; (N, M, D) & (N, 1, D)

        3 - array(s) with too few dims can have their
            shapes **prepended** with a dim of len 1
            to satisfy property 2
            A few examples:
        >>-----> (N, M, D) & (M, D); OKAY! ===> (M, D) --> (1, M, D)
                 (N, M, D) & (N, D);  BAD! ####  restore to (N,1,D)
                 (N, M, D) & (D,);   OKAY! ===> (1, 1, D)
                 (N, M, D) & (N,);    BAD! #### restore to (N,1,1)
                 (N, M, D) & (M,);    BAD! #### restore to (1,M,1)
                 (N, M, D) & ();     OKAY! ===> (1,1,1)
                                          (and scalars always broadcastable)
        Restoration func
        ================
        ASSUME: Property 1 is not a consideration here, given
                this func only called during backprop. Valid inputs
                will necessarily have reduced dimensions.

        RETURN: Y (the input), if property 2 is satisfied or
                Y with length 1 dims inserted until property 2 is satisfied
                * NB: Y rarely needs inserted dims, since most functions
                      preserve dimensions (keepdims=True)

        """
        # Get reduction flags
        axes, keep = self.flags

        # Check Property 2, or if Y scalar
        if keep or Y.ndim == 0:
            return Y

        # Insert dims until property 2 satisfied
        for ax in sorted(axes): # sort, to avoid deprecation warnings
            Y = np.expand_dims(Y, ax)
        return Y

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Sum(ReductionFunction): # #
    """ Compute sum along axis or axes """
    @staticmethod
    #def sum(x, axis=None, keepdims=False):
    def sum(x, **kwargs):
        return x.sum(**kwargs)

    @staticmethod
    def sum_prime(x, **kwargs):
        return np.ones(x.shape, x.dtype)

    def forward(self, X, axis=None, keepdims=False):
        #==== Cache inputs
        self.cache = X
        self.flags = self.format_axes(axis, X.shape), keepdims
        #==== sum X
        Y = self.sum(X, axis=axis, keepdims=keepdims)
        return Y

    def backward(self, gY):
        X = self.cache
        gX = self.restore_dims(gY) * self.sum_prime(X)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Mean(ReductionFunction): # #
    """ Compute mean along axis or axes """
    @staticmethod
    def mean(x, **kwargs):
        return x.mean(**kwargs)

    @staticmethod
    def mean_prime(x, axis=None):
        # div by number of elems averaged out in forward
        dims = x.shape
        num_avgd = np.prod([dims[i] for i in axis])
        return np.ones(x.shape, x.dtype) / float(num_avgd)

    def forward(self, X, axis=None, keepdims=False):
        #==== cache inputs
        self.cache = X
        self.flags = self.format_axes(axis, X.shape), keepdims
        #==== average X
        Y = self.mean(X, axis=axis, keepdims=keepdims)
        return Y

    def backward(self, gY):
        X = self.cache
        axes, keep = self.flags
        gX = self.restore_dims(gY) * self.mean_prime(X, axes)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Prod(ReductionFunction): # #
    """ Compute product along axis or axes """
    @staticmethod
    def prod(x, **kwargs):
        return x.prod(**kwargs)

    @staticmethod
    def prod_prime(x, axis=None):
        # Multiply where elems prod'd out in forward
        prod_elems = Prod.prod(x, axis=axis, keepdims=True) / x
        return np.ones(x.shape, x.dtype) * prod_elems

    def forward(self, X, axis=None, keepdims=False):
        #==== cache inputs
        self.cache = X
        self.flags = self.format_axes(axis, X.shape), keepdims
        #==== prod X
        Y = self.prod(X, axis=axis, keepdims=keepdims)
        return Y

    def backward(self, gY):
        X = self.cache
        axes, keep = self.flags
        gX = self.restore_dims(gY) * self.prod_prime(X, axes)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



#==============================================================================
#------------------------------------------------------------------------------
#                          Activation Functions
#------------------------------------------------------------------------------
#==============================================================================

class Sigmoid(Function): # #
    """ Logistic sigmoid activation """
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        y = Sigmoid.sigmoid(x)
        return y * (1 - y)

    def forward(self, X):
        self.cache = X
        Y = self.sigmoid(X)
        return Y

    def backward(self, gY):
        X = self.cache
        gX = gY * self.sigmoid_prime(X)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Tanh(Function): # #
    """ Hyperbolic tangent activation """
    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_prime(x):
        return -(np.square(np.tanh(x))) + 1

    def forward(self, X):
        self.cache = X
        Y = self.tanh(X)
        return Y

    def backward(self, gY):
        X = self.cache
        gX = gY * self.tanh_prime(X)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Softmax(Function): # #
    """ Softmax activation """
    @staticmethod
    def softmax(x):
        kw = {'axis':1, 'keepdims':True}
        exp_x = np.exp(x - x.max(**kw))
        return exp_x / np.sum(exp_x, **kw)

    @staticmethod
    def softmax_prime(x):
        kw = {'axis':1, 'keepdims':True}
        expx = np.exp(x - x.max(**kw))
        y = expx / np.sum(expx, **kw)
        return y - np.sum(np.square(y), **kw)

    def forward(self, X):
        self.cache = np.copy(X)
        Y = self.softmax(X)
        return Y

    def backward(self, gY):
        X = self.cache
        gX = gY * self.softmax_prime(X)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class ReLU(Function): #
    """ ReLU activation : zero out any elements < 0 """
    @staticmethod
    def relu(x):
        return x.clip(min=0)

    @staticmethod
    def relu_prime(x):
        return np.where(x < 0, 0, 1)

    def forward(self, X):
        self.cache = np.copy(X)
        Y = self.relu(X)
        return Y

    def backward(self, gY):
        X = self.cache
        gX = gY * self.relu_prime(X)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class SoftPlus(Function): #
    """ Softplus activation : smooth relu """
    @staticmethod
    def softplus(x):
        return np.log(1 + np.exp(x))

    @staticmethod
    def softplus_prime(x):
        exp_x = np.exp(x)
        return exp_x / (1 + exp_x)

    def forward(self, X):
        self.cache = np.copy(X)
        Y = self.softplus(X)
        return Y

    def backward(self, gY):
        X = self.cache
        gX = gY * self.softplus_prime(X)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class ELU(Function): #
    """ Exponential Linear Unit """
    def __init__(self, *args, alpha=1.0, **kwargs):
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    @staticmethod
    def elu(x, alpha):
        return np.where(x < 0, alpha * (np.exp(x) - 1), x)

    @staticmethod
    def elu_prime(x, alpha):
        return np.where(x < 0, alpha * np.exp(x), 1)

    def forward(self, X):
        self.cache = np.copy(X)
        Y = self.elu(X, self.alpha)
        return Y

    def backward(self, gY):
        X = self.cache
        gX = gY * self.elu_prime(X, self.alpha)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class SeLU(ELU): #
    """ Scaled Exponential Linear Units
    ELU, but with a scalar and finetuning

    SeLU is the activation function featured in
    "Self-Normalizing Neural Networks," and they
    are designed to implicitly normalize feed-forward
    networks.

    Reference
    ---------
    Paper : https://arxiv.org/abs/1706.02515
        explains the properties and derivations for SeLU
        parameters, but the precision values below from their
        project code @ github.com/bioinf-jku/SNNs/

    Parameters : alpha, scale
        github.com/bioinf-jku/SNNs/blob/master/getSELUparameters.ipynb
        github.com/bioinf-jku/SNNs/blob/master/selu.py
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    @staticmethod
    def selu(x, alpha, scale):
        return scale * ELU.elu(x, alpha)

    @staticmethod
    def selu_prime(x, alpha, scale):
        return scale * ELU.elu_prime(x, alpha)

    def forward(self, X):
        self.cache = np.copy(X)
        Y = self.selu(X, self.alpha, self.scale)
        return Y

    def backward(self, gY):
        X = self.cache
        gX = gY * self.selu_prime(X, self.alpha, self.scale)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Swish(Function): # #
    """ Self-gated activation function
    Can be viewed as a smooth function where the nonlinearity
    interpolates between the linear function (x/2), and the
    ReLU function.
    Intended to improve/replace the ReLU masterrace

    "The best discovered activation function",
    - Authors
    See: https://arxiv.org/abs/1710.05941

     Params
     ------
     X : ndarray
        the usual
     b : ndarray
        scaling parameter. Authors say it can be a
        constant scalar (1.0), but always immediately follow it up
        saying it's best as a channel-wise trainable param.

        So it's going to be always be treated as an array with the same
        shape as X.shape[1:] (exlude batch dim)
    """
    @staticmethod
    def swish(x, b):
        return x * Sigmoid.sigmoid(x * b)

    @staticmethod
    def swish_prime(x, b):
        #==== forward vars
        sig_xb = Sigmoid.sigmoid(x*b)
        y = x * sig_xb
        #==== diffs
        dx = sig_xb + b * y * (1 - sig_xb)
        db = y * (x - y)
        return dx, db

    def forward(self, X, b):
        self.cache = np.copy(X)
        Y = self.swish(X, b)
        return Y

    def backward(self, gY, B):
        X = self.cache
        dX, dB = self.swish_prime(X, B)
        gX = gY * dX
        gB = gY * dB
        return gX, gB.sum(axis=0)



#==============================================================================
#------------------------------------------------------------------------------
#                             Loss Functions
#------------------------------------------------------------------------------
#==============================================================================


#------------------------------------------------------------------------------
# Cross entropy
#------------------------------------------------------------------------------
class LogisticCrossEntropy(Function): #
    """ Logistic cross-entropy loss defined on sigmoid activation

    Truth labels are converted to one-hot encoding to reduce
    the duplicate select and reduction ops in forward and backprop

    """
    @staticmethod
    def logistic_cross_entropy(x, t):
        lhs = -(t * np.log(x))
        rhs = (1 - t) * np.log(1 - x)
        return np.mean(lhs - rhs)

    @staticmethod
    def logistic_cross_entropy_prime(x, t):
        return (x - t) / x.size

    def forward(self, X, t_vec):
        """
        Params
        ------
        X : ndarray.float32, (N, D)
            linear output of network's final layer
        t_vec : ndarray.int32, (N,) ----> (N, D)
            truth labels on each sample, converted from a 1D
            vector of vals within [0, D) to 2D 1-hot (with binary vals)

        Returns
        -------
        Y : float, (1,)
            average cross entropy error over all samples
        p : ndarray.int32, (N,D)
            Network approximations on class labels (for accuracy metrics)
        """
        # Process inputs
        #----------------------
        # Check dimensional integrity
        assert X.ndim == 2 and t_vec.shape[0] == X.shape[0]

        # Convert labels to 1-hot
        t = to_one_hot(np.copy(t_vec), X.shape[-1]) # (N,D)

        # Sigmoid activation
        #----------------------
        p = Sigmoid.sigmoid(np.copy(X))
        self.cache = p, t

        # Average cross-entropy
        #----------------------
        Y = self.logistic_cross_entropy(np.copy(p), t)
        return Y, p


    def backward(self, *args):
        """ Initial gradient to be chained through the network
        during backprop

        Params
        ------
        p : ndarray.float32, (N,D)
            sigmoid activation on network forward output
        t : ndarray.int32, (N,D), 1-hot
            ground truth labels for this sample set

        Returns
        -------
        gX : ndarray.float32, (N, D)
            derivative of X (network output) wrt the logistic loss

        """
        # Retrieve vars
        #----------------------
        p, t = self.cache

        # Get gradient
        #----------------------
        gX = self.logistic_cross_entropy_prime(p, t)
        return gX

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class SoftmaxCrossEntropy(Function): #
    """ Cross entropy loss defined on softmax activation

    Truth labels are converted to one-hot encoding to reduce
    the duplicate select and reduction ops in forward and backprop

    """
    @staticmethod
    def softmax_cross_entropy(x, t):
        tlog_x = -(t * np.log(x))
        return np.mean(np.sum(tlog_x, axis=1))

    @staticmethod
    def softmax_cross_entropy_prime(x, t):
        return (x - t) / x.shape[0]


    def forward(self, X, t_vec):
        """ Cross entropy loss function defined on a
        softmax activation

        Params
        ------
        X : ndarray.float32, (N, D)
            output of network's final layer with no activation. *2D assumed*
        t_vec : ndarray int32, (N,) ----> (N, D)
            truth labels for each sample, where int values range [0, D)
            converted to (N, D) one-hot encoding

        Returns
        -------
        Y : float, (1,)
            average cross entropy error over all samples
        p : ndarray.int32, (N,D)
            Network approximations on class labels (for accuracy metrics)
        """
        # Preprocess inputs
        #----------------------
        # Check dimensional integrity
        assert X.ndim == 2 and t_vec.shape[0] == X.shape[0]

        # Convert labels to 1-hot
        t = to_one_hot(np.copy(t_vec)) # (N,D)

        # Softmax activation
        #----------------------
        p = Softmax.softmax(X)
        self.cache = p, t

        # Average cross entropy
        #----------------------
        Y = self.softmax_cross_entropy(np.copy(p), t)
        return Y, p


    def backward(self, *args):
        """ Initial backprop gradient grad on X wrt loss

        Params
        ------
        p : ndarray.float32, (N,D)
            activation on network forward output
        t : ndarray.int32, (N,D), 1-hot
            ground truth labels for this sample set

        Returns
        -------
        gX : ndarray, (N, D)
            derivative of X (network prediction) wrt the cross entropy loss
        """

        # Retrieve data
        p, t = self.cache

        # Calculate grad
        #---------------
        gX = self.softmax_cross_entropy_prime(p, t)
        return gX

#------------------------------------------------------------------------------
# Distance
#------------------------------------------------------------------------------

class MSE(ReductionFunction): #
    """ Mean Squared Error loss function 
    """
    @staticmethod
    def mse(x, y, **kwargs): #sum_axis=None): # not yet supported
        squared_error = Square.square(x - y)
        z = Mean.mean(squared_error, **kwargs)
        return z

    @staticmethod
    def mse_prime(x, y, axis=None):
        #==== inputs
        err = x - y
        se = Square.square(err)
        #==== derivs
        dz = Mean.mean_prime(se, axis)
        dx = 2 * err * dz
        dy = -dx
        return dx, dy

    def forward(self, X, Y, axis=None, keepdims=False):
        #==== cache inputs
        self.cache = X, Y
        self.flags = self.format_axes(axis, X.shape), keepdims
        #==== calculate mse
        Z = self.mse(X, Y, axis=axis, keepdims=keepdims)
        return Z

    def backward(self, *args):
        X, Y = self.cache
        axes, keep = self.flags
        gX, gY = self.mse_prime(X, Y, axes)
        return gX



#==============================================================================
#                        Regularization and Normalization Functions
#==============================================================================

class Dropout(Function):
    """ Randomly drops input elements with the probability given by ratio

    Dropout will zero out elements of the input at random with
    probability ratio p, and then scale the remaining elements by
    by 1 / (1 - p).

    Dropout is a well established technique commonly used in feedforward
    networks in order to reduce overfitting on a training set.
    The idea is that network connections, instead of learning features by the
    detection and context of others, the connections are instead encouraged
    to learn more robust detection of features.
    """
    def __init__(self, drop_rate=0.5):
        """ 50% drop-rate is suggested default """
        self.drop_rate = drop_rate
        super().__init__()

    def get_mask(self, X):
        if hasattr(self, 'mask'): return self.mask
        p  = self.drop_rate
        drops = np.random.rand(*X.shape) >= p
        scale = 1 / (1 - p)
        self.mask = (drops * scale).astype(X.dtype)
        return self.mask

    @staticmethod
    def dropout(x, mask):
        return x * mask

    @staticmethod
    def dropout_prime(x, mask):
        return mask

    def forward(self, X):
        if self.test: return X # don't drop test elements
        mask = self.get_mask(X)
        Y = self.dropout(X, mask)
        return Y

    def backward(self, gY):
        mask = self.mask
        gX = gY * mask
        return gX




#==============================================================================
#------------------------------------------------------------------------------
#                            FUNCTION COLLECTIONS
#------------------------------------------------------------------------------
#==============================================================================
ACTIVATIONS = {'sigmoid'  : Sigmoid,
                'tanh'    : Tanh,
                'softmax' : Softmax,
                'relu'    : ReLU,
                'softplus': SoftPlus,
                'elu'     : ELU,
                'selu'    : SeLU,
                'swish'   : Swish,
                }

CONNECTIONS = {'bias'  : Bias,
               'matmul': MatMul,
               'linear': Linear,
               #'lstm' : LSTM,
               }

OBJECTIVES = {'logistic_cross_entropy': LogisticCrossEntropy,
              'softmax_cross_entropy' :  SoftmaxCrossEntropy,
              }






"""===========================================================================#

STILL PENDING REWORK FROM v1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[X] ReductionFunction
    [X] Sum
    [X] Mean
    [X] Prod
    [ ] Max/Min


IN PIPELINE:
LSTM : stub and gradient reference
    just walk it back now and consolidate ops; probably only need to cache
    3~4 things on forward
      - *even still* this likely won't work as a Function in the same manner
         as the other connection func Linear does. Having to keep cell-state
         and previous output, in addition to normal caching, makes it make
         more sense as a layer

LayerNorm : working on stdv
    the actual layernorm func is simple, but requires standard dev func;
    should be able to just compose the related functions (mean, square, sqrt)
    to make a stdv function class, then use that for layernorm func

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""


'''
#==============================================================================
#------------------------------------------------------------------------------
#                            CONNECTION FUNCTIONS
#------------------------------------------------------------------------------
#==============================================================================


# NAIVE, BY-HAND GRADIENT VERSION
@TODO

# WIP
class LSTM(Function):
    """ Stateless Long short-term memory function

    A LSTM is a type of recurrent network unit defined as block of
    gated operations on external and internal representations.

    It takes in some external input X, which is typically a sample within
    a sequence, along with weights W, and performs a series of gated
    operations on these inputs combined with it's own stateful variable
    (cell-state) C, and "memory" the previous ouput in the sequence H.

    Params
    ------
    X : ndarray, (N, D)
        external input to unit

    W : ndarray, (D+K, 4*K)
        the weight variables for each of the four gated operations (hence
        the 4*K output channel)
        These 4 gates are as follows:
        i : input gate
        a : activation gate
        f : forget gate
        o : output gate

    H : ndarray, (N, K)
        previous function output

    C : ndarray, (N, K)
        cell state

    While the gated operations are typically computed in combination with
    constituent weight variable, the individual weights are instead
    concatenated, and matmul against the concatenated X, H
    """
    @staticmethod
    def lstm(x, h, c, w):
        pass

    def forward(self, X, W, H, C):
        cache = []
        # Concat input and previous output
        Z = np.concatenate([X, H], axis=1) # (N, D+K)
        cache.append(Z)

        # Transform and split into units
        #-------------------------------
        iafo = np.matmul(Z, W) # (N, D+K).(D+K, 4*K)
        i, a, f, o = np.split(iafo, 4, axis=1) # (N, K)

        # Gate each unit
        #---------------
        ai = Sigmoid.sigmoid(i)  # ai --> "[a]ctivated [i]nput" ;]
        aa = Tanh.tanh(a)
        af = Sigmoid.sigmoid(1 + f)
        ao = Sigmoid.sigmoid(o)

        # Cache for backprop
        cache.extend([ai, aa, af, ao, C])

        # Update cell-state
        #------------------
        C_t = ai * aa + af * C
        aC_t = Tanh.tanh(np.copy(C_t))

        # Cache for backprop
        cache.append(aC_t)
        self.cache = cache

        # Calculate output
        H_t = ao * aC_t
        return H_t, C_t

    def backward(self, gH, W):
        # Retrieve intermediate vars from cache
        Z, ai, aa, af, ao, C, aC_t = self.cache

        # Backprop through gated ops
        #---------------------------
        #==== H_t
        do = gH * aC_t * Sigmoid.sigmoid_prime(ao)
        dC = gH * ao * Tanh.tanh_prime(aC_t)
        #==== C_t
        di = dC * aa * Sigmoid.sigmoid_prime(ai)
        da = dC * ai * Tanh.tanh_prime(aa)
        df = dC * C  * Sigmoid.sigmoid_prime(af)
        gC = dC + af

        # Backprop through transformation
        #--------------------------------
        diafo = np.concatenate([di, da, df, do], axis=1) # (N, 4*K)
        dZ = np.matmul(diafo, W.T) # (N, 4*K).(4*K, D+K)
        gW = np.matmul(Z.T, diago) # (D*K, N).(N, 4*K)

        # Split inputs
        gX, gH = np.split(dZ, [-C.shape[-1]], axis=1)
        return gX, gW, gH, gC

     def __call__(self, *args, **kwargs):
        print('LSTM WIP, PENDING TESTING, DO NOT CALL')
        assert False

'''




#------------------------------------------------------------------------------
# Normalization
#------------------------------------------------------------------------------
'''
class LayerNormalization(Function):
    def __init__(self, *args, **kwargs):
        self.mean = Mean()
        self.sqrt = Sqrt()
        self.square = Square()
        super().__init__(*args, **kwargs)

    @staticmethod
    def layer_normalization(x, g, b, std, mu):
        y = (g / std) * (x - mu) + b
        return y

    @staticmethod
    def layer_normalization_prime(x, g, b, dstd, dmu):
        pass




class LayerNormalization(Function):
    """ Normalization routine defined on statistics from the summed
        connections between a layer and a single input case
    It is designed to overcome several drawbacks of batch normalization,
    the most significant being the latter's dependence on batch-size

    # see p.3 of orig article for func defs
    #-----> https://arxiv.org/pdf/1607.06450.pdf


    # by hand
    x : (N, D)
    g : (D,)
    b : (D,)
    #---------

    #===== Fwd
    mu = np.mean(x, **kw) # (N, 1)
    diff = x - mu
    sqr_diff = diff**2
    var = np.mean(sqr_diff, **kw) # (N, 1)
    std = np.sqrt(var) # (N, 1)
    #- - - - - - -
    gds = g / std  # (N, D)
    xmu = x - mu   # (N, D)
    gdxm = gds * xmu
    y = gdxm + b # (N, D)

    #===== Bwd
    dgdxm = gY
    db = gY.sum(0)
    dgds = dgdxm * xmu
    dxmu = dgdxm * gds
    dx  = dxmu
    dmu = -dxmu
    dg = dgds * -(1/np.square(std)) # or -1 / var
    dstd = dgds * g

    dvar = dstd * 1/(2*sqrt(var))
    dsqr_diff = dvar * np.broadcast_to(sqr_diff, (x.shape)) / x.shape[1]
    ddiff = dsqr_diff * 2*diff
    dx += ddiff
    dmu += -dxmu
    dx += dmu * np.broadcast_to(mu, x.shape) / x.shape[1]


    @staticmethod
    def get_stats(x):
        kw = {'axis':1, 'keepdims':True}
        mu  = np.mean(x, **kw)
        std = np.std(x, **kw) # (N, 1)
        return mu, std

    @staticmethod
    def layer_norm(x, g, b):
        mu, std = LayerNormalization.get_stats(np.copy(x))
        y = (g / std) * (x - mu) + b
        return y

    @staticmethod
    def layer_norm_prime(x, g, b):
        pass


    def forward(self, X, gain, bias):
        self.cache = X, gain, bias
        y = self.layer_norm(X, gain, bias)
        return y

    def backward(self, gY):
        pass
        """

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class MaxMin(ReductionFunction):
    """ Base class for max, min funcs """
    MM_func  = None
    cmp_func = None

    def reset_fn_vars(self):
        super().reset_fn_vars()
        self.X = self.Y = None

    def forward(self, X, axis=None, keepdims=False):
        self.fn_vars = X.shape, axis
        Y = self.MM_func(X, axis=axis, keepdims=keepdims)
        self.X = X
        self.Y = Y
        return Y

    def backward(self, gY):
        X  = self.X
        Y  = self.restore_shape(self.Y)
        gY = self.restore_shape(gY)
        gX = np.where(self.cmp_func(X, Y), gY, 0)
        self.reset_fn_vars()
        return gX


class Max(MaxMin):
    """ Computes max along axis """
    MM_func  = np.max
    cmp_func = lambda x, y: x >= y

class Min(MaxMin):
    """ Computes min along axis """
    MM_func  = np.min
    cmp_func = lambda x, y: x < y

'''



