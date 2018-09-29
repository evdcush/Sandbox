"""
The layers that compose the neural network are in this module.

Layers serve as the interface to Functions for the Network.
They manage any parameters used by the functions, and provide high
level abstractions for backprop and updates.

There are (currently) two kinds of layers: parametric and static

# Layer types
#------------
Parametric layers : [Dense, Swish, ]
    these layers use variables that receive updates from an optimizer

Static layers : [Sigmoid, Tanh, Softmax, ReLU, ELU, SELU, ]
    these layers do not utilize variables and do not
    need updates from optimizer

"""
import code
import numpy as np
import functions
from initializers import HeNormal, Zeros, Ones

#==============================================================================
#------------------------------------------------------------------------------
#                          Parametric Layers
#------------------------------------------------------------------------------
#==============================================================================
""" These layers all have parameters that are updated through gradient descent
"""

#class ParametricLayer:
#    """ Abstracts some of the boilerplate from most parametric layers """
#    __slots__ = ('name', 'func', 'updates', 'params', key)


class Dense:
    """ Vanilla fully-connected hidden layer
    The Dense layer is defined by the linear transformation function:

         f(X) = X.W + b
    Where
    : W is a weight matrix
    : b is a bias vector,
    and both are learnable parameters optimized through gradient descent.

    Attributes
    ----------
    updates : bool
        whether the layer has learnable parameters
    name : str
        layer name, as class-name + ID
    ID : int
        Dense layer instance's position in the network
    kdims : tuple(int)
        channel sizes (determines dimensions of params)
    linear : Function.Linear
        linear function instance, which performs the Y = X.W + B function
    W_Key : str
        unique string identifier for W
    B_key : str
        unique string identifier for B

    Params
    ------
    W : ndarray.float32, of shape kdims
        weight matrix
    B : ndarray.float32, of shape kdims[-1]
        bias vector

    """
    updates = True

    def __init__(self, ID, kdims, init_W=HeNormal, init_B=Zeros):
        self.name = str(self.__class__.__name__) + ID
        self.ID = ID
        self.kdims = kdims
        self.linear = functions.Linear()
        self.initialize_params(init_W, init_B)

    def __repr__(self):
        # eval repr format:
        rep = "{}('{}, {}')"
        cls_name = self.__class__.__name__
        kdims = self.kdims
        ID = self.ID

        # Format eval repr and ret
        rep_args = (cls_name, ID, kdims, self.nobias)
        return rep.format(*rep_args)

    # Layer parameter initialization
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def initialize_params(self, init_W, init_B):
        """ Initializes the parameters for the Dense layer instance

        This function both can initialize new variables for the
        parameters, as well as restore pretrained variables,
        based on the `type` of init_W/init_B. Both variables
        are restored as a pair.

        If the init args are of type ndarray, than they are
        pretrained (or preinitialized) variables. Otherwise,
        they are Initializers.

        Params
        ------
        init_W : Initializer OR ndarray
            initializes the starting values for the weight matrix
        init_B : Initializer OR ndarray
            initializes the starting values for the bias
        """
        # Layer dims
        w_dims = self.kdims
        b_dims = self.kdims[-1:]

        # Var keys
        self.W_Key = self.name + 'W'
        self.B_key = self.name + 'B'

        # Check whether inits are arrays
        if isinstance(init_W, np.ndarray):
            # Check dimensional integrity
            assert (init_W.shape == w_dims) and (init_B.shape == b_dims)
            self.W = init_W
            self.B = init_B
        else:
            # Initializer instances
            w_init = init_W()
            b_init = init_B()
            # Initialize vars
            self.W = w_init(w_dims)
            self.B = b_init(b_dims)

        # Gradient placeholders
        self.W_grad = None
        self.B_grad = None


    # Layer optimization
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def update(self, opt):
        """ Update weights and bias with gradients from backprop
        Params
        ------
        opt : Optimizer instance
            Optimizing algorithm, updates through __call__
        """
        # Make sure grads exist
        assert self.W_grad is not None and self.B_grad is not None

        # Group vars with their grads, keyed to their respective keys
        params = {}
        params[self.W_Key] = (self.W, self.W_grad)
        params[self.B_Key] = (self.B, self.B_grad)

        # Get updates
        updated_params = opt(params)
        self.W = updated_params[self.W_key]
        self.B = updated_params[self.B_key]

        # Reset grads
        self.W_grad = None
        self.B_grad = None

    # Layer network ops
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def forward(self, X):
        W = self.W
        B = self.B
        Y  = self.linear(X, W, B)
        return Y

    def backward(self, gY):
        # Backprop
        W = self.W
        B = self.B
        gX, gW, gB = self.linear(gY, W, B, backprop=True)

        # Assign var grads and chain gX to next layer
        self.W_grad = gW
        self.B_grad = gB
        return gX

#==============================================================================
#==============================================================================



#==============================================================================
#==============================================================================

# Available Layers
#-----------------
OPS = {'dense_layer': Dense,}

ACTIVATIONS = {'sigmoid_layer' : SigmoidActivation,
               'softmax_layer' : SoftmaxActivation,
                  'tanh_layer' : TanhActivation,
                  'relu_layer' : ReLUActivation,
                   'elu_layer' : ELUActivation,
                  'selu_layer' : SeLUActivation,
                  'swish_layer': SwishActivation,
                  }

LAYERS = {**OPS, **ACTIVATIONS}

def get_all_layers():
    return LAYERS
