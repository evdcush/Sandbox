"""
Neural network implementations are located in this module.

# Neural networks
#=================================

Basic process
-------------
Neural networks are composed of Layers
For our usage, we can think of the network both as an algorithm
and as a sort of ordered "data-structure," like a list or array,
and Layers being the elements of that list.

    # Forward
      - To generate a prediction, we iterate through the "list" in order,
        propagating an initial input through each element.

    # Backward
      - To optimize the network's predictions, we iterate through the list
        in reverse order, propagating the gradient of an objective
        function through each element.

Control Flow
------------
1 - Network instance initialized with a list of channels
    and Layer-types
2 - Network receives external data input X, and propagates9
    it through it's Layers
3 - A final output Layer returns a prediction
  - (error between network prediction and truth is evaluated by an objective)
4 - If training: the network receives the gradient of a loss
    function and backpropagates the gradient through its
    layers (in reverse order)
    - If the layer has learnable parameters, such as a Dense layer,
      it will store the gradients for its variables
5 - An optimizer then updates all learnable variables in the network's
    layers

* This process is repeated indefinitely, typically until it has converged
  upon the best local minimum, or whenever the specified number of epochs
  or iterations has been reached


Available implementations
-------------------------
NeuralNetwork : fully-connected, feed-forward network
    - The fully-connected, feed-forward  neural network, is one of the most
      elementary types of networks.

    - It is composed of fully-connected layers that perform
      linear-transformations on input data, that are then 'activated'
      by nonlinear functions, such as the logistic function or
      hyperbolic tangent.

    - Typically shallower than other types of networks (though this
      implementation is of arbitrary depth)
"""
import numpy as np
import layers as L
import functions as F



class NeuralNetwork:
    """ Fully-connected, feed-forward neural network """
    def __init__(self, channels, activation=F.SeLU, use_dropout=True):
        """ Initializes an arbitrarily deep neural network
        Params
        ------
        channels : list(int)
            channel sizes for each layer; determines the dimensionality
            of the layer transformations, as well as depth of network
        activation : Function OR ParametricLayer
            activation function, or layer if parameterized
        use_dropout : bool
            whether to use dropout function
        """
        self.channels = list(zip(channels, channels[1:])) # tuple (k_in, k_out)
        self.activation  = activation
        self.use_dropout = use_dropout

        # Initialize layers
        #------------------
        self.layers = []
        for ID, kdims in enumerate(self.channels, 1):
            last_layer = ID == len(self.channels)

            #==== Connection
            connection = L.Dense(ID, kdims)
            self.layers.append(connection)

            if not last_layer:
                #==== Dropout
                if use_dropout:
                    dropout = F.Dropout()
                    self.layers.append(dropout)

                #==== Activation
                activation = self.activation(ID, kdims)
                self.layers.append(activation)


    # Network algorithm
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def forward(self, X, test=False):
        """ Propagates input X through network layers """
        Y = np.copy(X)
        for layer in self.layers:
            Y = layer(Y, test=test)
        return Y

    def backward(self, gY):
        """ Backpropagation through layers
        Params
        ------
        gY : ndarray
            gradient of loss function wrt to network output Y (or Y_hat)
        """
        gX = np.copy(gY)
        for layer in reversed(self.layers):
            gX = layer(gX, backprop=True)

    #==== Optimizer update
    def update(self, opt):
        """ Pass optimizer through parametric units of layers """
        for unit in self.layers:
            if unit.__module__ == 'layers':
                unit.update(opt)

    # Naming formats
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def __str__(self):
        name  = self.__class__.__name__
        return name

    def __repr__(self):
        # eval-style repr
        rep = ("{}('{}, activation={}, use_dropout={}')")
        name  = self.__class__.__name__

        #==== Get instance vars
        chans = self.channels
        act   = self.activation
        drop  = self.use_dropout

        #==== format repr and return
        rep = rep.format(name, chans, act, drop)
        return rep
