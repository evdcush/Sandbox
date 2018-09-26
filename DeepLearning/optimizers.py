""" This module contains all optimization routines
available to the model.

Currently only SGD and Adam are available.


# Module components
#------------------
Optimizer : base class for gradient-based optimizers
    Receives a parameter, the gradient wrt to that parameter
    via a stochastic objective function, and updates the
    parameter through some routine

AdaptiveOptimizer : base class for momentum and accelerated optimizers
    Typically has a weight vector for each learnable/updatable parameter
    based on that parameter's gradient history

SGD : Optimizer, vanilla stochastic gradient descent algorithm
    Optimizes a parameter based on the gradients of a small
    subset (minibatch) of training data and a learning rate

Adam: Optimizer, Adaptive moment estimation algorithm
    Optimizes a parameter adaptively based on estimates
    from its previous, decaying, average gradients
    squared v, and and average gradients m

"""
import code
import numpy as np

from utils import TODO, NOTIMPLEMENTED, INSPECT

""" module imports
utils :
    `TODO` : decorator
        serves as comment and call safety

    `NOTIMPLEMENTED` : decorator
        raises NotImplementedErrorforces if class func has not been overridden

    `INSPECT` : decorator
        interrupts computation and enters interactive shell,
        where the user can evaluate the input and output to func
"""
#==============================================================================
#------------------------------------------------------------------------------
#                              Functions
#------------------------------------------------------------------------------
#==============================================================================
#==============================================================================
# Globals
#==============================================================================
#------------------------------------------------------------------------------
# Shift invariant ops
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Shift invariant nodes
# ========================================

#==== Thing

# Thing
#-----------

#==============================================================================
#------------------------------------------------------------------------------
#                              Optimizers
#------------------------------------------------------------------------------
#==============================================================================

#==============================================================================
# Base Optimizer classes :
#  Optimizer, AdaptiveOptimizer
#==============================================================================

# Optimizer
# ---------
# inherits :
# derives : AdaptiveOptimizer, SGD
class Optimizer:
    """ Base optimizer
    Uses gradient descent to move towards (local) minimas in the
    space of an objective (or loss) function

    Params
    ------
    lr : float
        learning-rate for optimizer, typically a small
        float value within [0.1, 1e-4], serves as the
        "step-size" during gradient descent in terms of
        how far we want to move towards the opposite of
        gradient. The greater the value, the greater
        the updates to a param.

    """

    def __init__(self, lr, *args, **kwargs):
        self.lr = lr

    def update(self, P, gP):
        return P - self.lr * gP

    def __call__(self, params, grads):
        """ Adjust parameters based on gradient from objective

        Params
        ------
        params : dict(str : ndarray)
            parameters to receive updates
        grads : dict(str : ndarray)
            gradients of objective/loss function wrt the
            parameters

        """
        updated_params = {}
        for p_key in params.keys():
            # get param and it's respective grad
            P = params[p_key]
            gP = grads[p_key]

            # update param
            updated_params[p_key] = self.update_param(P, gP)
        return updated_params

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# AdaptiveOptimizer
# -----------------
# inherits : Optimizer
# derives : Adam
class AdaptiveOptimizer(Optimizer):
    """ Base adaptive optimizer

    In addition to a learning-rate, parameters updated by
    an adaptive optimizer an AdaptiveOptimizer are also adjusted
    by some parameter-specific variable, often a "momentum" vector,
    based on that parameter's past gradient values

    Attributes
    ----------
    moments : dict(ndarray)
        collection of moment vectors keyed to a param in the model

    Params
    ------
    momentum : float
        the decay rate on past gradient values

    moments_init : dtype(moments)
        the pretrained or saved moments from another model
        to be restored

    """
    moments = {}

    def __init__(self, lr, *args, momentum=0.9, moments_init=None, **kwargs):
        self.lr = lr
        self.momentum = momentum

        if moments_init is not None:
            self.moments = moments_init

    @NOTIMPLEMENTED
    def initialize_moments(self, P):
        pass

    @NOTIMPLEMENTED
    def update(self, P, gP, P_key):
        pass

    def __call__(self, params, grads):
        """ Adjust parameters based on gradient from objective

        Params
        ------
        params : dict(str : ndarray)
            parameters to receive updates
        grads : dict(str : ndarray)
            gradients of objective/loss function wrt the
            parameters

        """
        updated_params = {}
        for p_key in params.keys():
            # get param and it's respective grad
            code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
            P = params[p_key]
            gP = grads[p_key]

            # update param
            updated_params[p_key] = self.update(P, gP, p_key)

        return updated_params

#==============================================================================
# Optimizers :
#  SGD, Adam
#==============================================================================

class SGD(Optimizer):
    """ Vanilla stochastic gradient descent

    Already fully implemented by base class
    """
    pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Adam(AdaptiveOptimizer):
    """ Adaptive optimization algorithm for gradient descent

    Adam uses adaptive estimates of lower-order moments,
    mean and variance ('m' and 'v', respectively) to optimize
    an objective

    Attributes
    ----------
    moments : dict(str : dict(str : ndarray))
        Collection of moment vectors keyed to a param in the model.
        For each parameter, there is a dict with two ndarrays,
        'm' and 'v', which are the moments for that param

    t : int
        timestep corresponding to number of updates made
        (ie, number of epochs or iterations completed thus far),
        used adapting stepsize (learning rate) for each update

    Params
    ------
    alpha : float
        stepsize
    beta1 : float
        exponential decay rate for first-order moment ('m')
    beta2 : float
        exponential decay rate for second-order moment ('v')
    eps : float
        arbitrarily small value used to prevent division by zero

    """
    moments = {} # eg, moments['layer2_W1'] = {'m': ndarray, 'v': ndarray}
    t = 0 # timestep

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                 moments_init=None):
        """ suggested default values (by authors) """
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps

        # restore pretrained moments
        if moments_init is not None:
            self.moments = moments_init


    def init_moments(self, P, P_key):
        """ initialize Adam moment estimates m, v for param P"""
        m = np.zeros_like(P)
        v = np.zeros_like(P)
        self.update_moments(P_key, m, v)
        return self.moments[P_key]

    def update_moments(self, P_key, m, v):
        """ Update Adam moment estimates m, v """
        updated_moments = {'m': m, 'v': v}
        self.moments[P_key] = updated_moments

    def get_moments(self, P, P_key):
        """ get moment estimates from collection
        If moments for parameter P do not exist, first initialize
        """
        if P_key not in self.moments:
            moments = self.init_moments(P, P_key)
        else:
            moments = self.moments[P_key]
        return moments


    @property
    def step(self):
        """ calculate current stepsize based on bias-corrected
        decay rates and timestep (Performed outside the update
        body for efficiency)
        """
        # Get params
        #-----------
        alpha = self.alpha
        beta1 = self.beta1
        beta2 = self.beta2
        t = self.t

        # bias correction func
        correct = lambda beta: 1 - np.power(beta, t)

        # Bias corrections
        #-----------------
        b1 = correct(beta1)
        b2 = correct(beta2)

        # Step at time t
        #---------------
        step = alpha * np.sqrt(b2) / b1
        return step


    def update(self, P, gP, P_key):
        """ Update parameter P with gradient gP """

        # update timestep
        self.t += 1

        # Get Adam update params
        #-----------------------
        step  = self.step
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.eps

        # Get moments
        #------------
        P_moments = self.get_moments(P, P_key)
        m = P_moments['m']
        v = P_moments['v']

        # Update moments
        #---------------
        m = beta1 * m + (1 - beta1) * gP
        v = beta2 * v + (1 - beta2) * np.square(gP)
        self.update_moments(P_key, m, v)

        # Update param P
        #---------------
        P_update = P - step * m / (np.sqrt(v) + eps)
        return P_update