"""
The layers that compose the neural network are in this module.

# Layers as an interface
The Layers class provide an interface for the network and optimizer
to functions that use learnable parameters.
    These include the basic connection functions, such as functions.Linear,
    as well as non-connection functions that use optimizable parameters,
    such as functions.Swish

Layers also allows their respective functions to remain strictly
functional, managing the parameters and gradients (and the
constituent initialization and updating).

"""
import functions
from initializers import HeNormal, Zeros, Ones

#==== Initializers
#HeNormal = initializers.HeNormal
#Zeros    = initializers.Zeros
#Ones     = initializers.Ones

#==============================================================================
#------------------------------------------------------------------------------
#                                Layers
#------------------------------------------------------------------------------
#==============================================================================


class Layer:
    """ Base class defining the structure and representation of layers

    Attributes
    ----------
    name : str
        layer name, as class-name + ID
    ID : int
        layer instance position in the network
    kdims : tuple(int); #---> (num_input_nodes, num_output_nodes)
        the dimensions for this layer, which determines the shape
        of the layer parameters

    """
    def __init__(self, ID, kdims, **kwargs):
        self.name = '{}-{}'.format(self.__class__.__name__, ID)
        self.ID = ID
        self.kdims = kdims

    def __str__(self,):
        # str : Dense-$ID;  (eg 'Dense-3')
        return self.name

    def __repr__(self):
        # repr : Dense($ID, $kdims); (eg "Dense('3, (32, 16)')")
        ID = self.ID
        kdims = self.kdims
        cls_name = self.__class__.__name__

        # Formal repr to eval form
        rep = "{}('{}, {}')".format(cls_name, ID, kdims)
        return rep

    def initialize_params(self, layer_vars):
        """ Initializes optimizable layer_vars for a Layer instance

        Params
        ------
        layer_vars : list(dict)
            dict of form: {'tag': str, 'dims': tuple(int), 'init': Initializer}
                tag  : unique name for the parameter
                dims : kdims (dimensions or shape of param)
                init : Initializer to be used
        """
        key_val = '{}_{{}}'.format(self.name)
        for layer_var in layer_vars:
            # Unpack var attributes
            #-----------------------------
            tag = layer_var['tag']
            dims = layer_var['dims']
            initializer = layer_var['init']()

            # Initialize instance attrs
            #------------------------------
            # var key
            setattr(self, '{}_key'.format(tag), key_val.format(tag))

            # variable
            setattr(self, tag, initializer(dims))

            # variable grad placeholder
            setattr(self, '{}_grad'.format(tag), None)


    # Layer optimization
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def update(self, opt):
        """ Update parameters with gradients from backprop

        Params
        ------
        opt : Optimizer instance
            Optimizing algorithm, updates through __call__
        """
        raise NotImplementedError

    def __call__(self, *args, backprop=False, test=False):
        func = self.backward if backprop else self.forward
        return func(*args, test=False)


#==============================================================================

class Dense(Layer):
    """ Vanilla fully-connected hidden layer
    Dense provides the interface to functions.Linear, maintaining the
    connection variables W, and B, in the function:
        f(X) = X.W + B
    Where
      X : input matrix
      W : weight matrix
      B : bias vector
    and both are learnable parameters optimized through gradient descent.

    Attributes
    ----------
    linear : functions.Linear
        linear function instance, which performs the Y = X.W + B function
    W_key : str
        unique string identifier for W
    B_key : str
        unique string identifier for B
    W : ndarray, of shape kdims
        weights parameter
    B : ndarray, of shape kdims[-1]
        bias parameter
    W_grad, B_grad : None | ndarray
        gradient placeholders for their respective parameters

    """
    def __init__(self, ID, kdims, init_W=HeNormal, init_B=Zeros):
        super().__init__(ID, kdims)
        self.linear = functions.Linear()
        self.initialize_vars(init_W, init_B)

    def initialize_vars(self, init_W, init_B):
        var_features = [{'tag': 'W', 'dims': self.kdims, 'init': init_W},
                        {'tag': 'B', 'dims': self.kdims[-1:], 'init': init_B},]
        super().initialize_params(var_features)

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
        params[self.W_key] = (self.W, self.W_grad)
        params[self.B_key] = (self.B, self.B_grad)

        # Get updates
        updated_params = opt(params)
        self.W = updated_params[self.W_key]
        self.B = updated_params[self.B_key]

        # Reset grads
        self.W_grad = None
        self.B_grad = None

    # Layer network ops
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def forward(self, X, **kwargs):
        # Outputs
        W = self.W
        B = self.B
        Y = self.linear(X, W, B, **kwargs)
        return Y

    def backward(self, gY, **kwargs):
        # Grads
        W = self.W
        B = self.B
        gX, gW, gB = self.linear(gY, W, B, backprop=True)

        # Assign var grads and chain gX to next layer
        self.W_grad = gW
        self.B_grad = gB
        return gX

#==============================================================================

class Swish(Layer):
    """ Swish activation layer """
    def __init__(self, ID, kdims, init_B=Ones, **kwargs):
        super().__init__(ID, kdims, **kwargs)
        self.swish = functions.Swish()
        self.initialize_vars(init_B)

    def initialize_vars(self, init_B):
        var_features = [{'tag': 'B', 'dims': self.kdims[-1:], 'init': init_B},]
        super().initialize_params(var_features)

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
        assert self.B_grad is not None

        # Group vars with their grads, keyed to their respective keys
        params = {}
        params[self.B_key] = (self.B, self.B_grad)

        # Get updates
        updated_params = opt(params)
        self.B = updated_params[self.B_key]
        self.B_grad = None

    # Layer network ops
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def forward(self, X, **kwargs):
        # Outputs
        B = self.B
        Y = self.swish(X, B, **kwargs)
        return Y

    def backward(self, gY, **kwargs):
        # Grads
        B = self.B
        gX, gB = self.swish(gY, B, backprop=True)

        # Assign var grads and chain gX to next layer
        self.B_grad = gB
        return gX


#==============================================================================

# Available Layers
#-----------------
CONNECTIONS = {'dense': Dense,}

# Special cases
#-----------------
PARAMETRIC_FUNCTIONS = {'swish': Swish}
LAYERS = {**CONNECTIONS, **PARAMETRIC_FUNCTIONS}


#==============================================================================
