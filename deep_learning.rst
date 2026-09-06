#############
Deep learning
#############
Simple and (mostly) clean implementations of various neural network stuff.

Current deep learning implementations are a simple feed-forward network.

Structure
=========
The project codebase, located in the |deep_learning directory|_, is split neatly into different modules.

:functions: Contains all functions related to the computation of the network. Activations, transformations, loss functions, regularizers--they are all located in `functions.py`_.
:layers: A broadly used term, layers are the essential component or architectural feature of neural networks. Layers, or "hidden layers," in this project refer specifically to the units or modules that *perform transformations on the data with a learnable or optimizable set of weights.* These are also referred to as "connections" in docstrings and comments within the code. All layers are are located in `layers.py`_.

  NB: Some functions, such as the ``Swish`` activation, have optimizable parameters, and corresponding ``Layer`` wrappers in `layers.py`_.

:network: The core component of our classifier model, all neural networks are implemented in the `network.py`_ module. Currently, the only available network implementation is a fully-connected feed-forward network with ``Dense`` layers.
:optimizer: The optimization routines for the neural network are implemented in this module. Available optimizers are ``SGD``--vanilla Stochastic Gradient Descent, and ``Adam``, a popular and powerful adaptive optimization algorithm based on moment estimation. Optimizers are located in `optimizers.py`_.

:train: The training routine is essentially an interface to all the modules of the project. It instantiates the network, and its layers by extension, the optimizer, and objective function, and loads the dataset through ``utils``. That being said, the majority of heavy lifting, such parameter initialization, file R/W, network ops, session information, etc. have been extracted to their constituent modules. Training especially relies on the utilities found in `utils.py`_. There is also a Trainer class within `utils.py`_ that performs the main operations as the train script, but has fewer options.

    Note that testing, or evalation *also* takes place within the `train.py`_ script.

:initializers: All learnable network parameters are initialized by the various initialization functions located in `initializers.py`_. Glorot normal, He normal, uniform, and constant initializations are available to choose from.
:utils: `utils.py`_ contains all functions not strictly related to network ops or computation. Something of a do-all script, utils contains the `Parser` class, and specifies the default network configuration (and updates it based on parsed command line args), maintains all information and statistics about training/testing error and accuracy, and provides various specialied preprocessing operations on data, such as one-hot encodings and the accuracy function.


Learning tasks
--------------
The deep learning project is defined on the `Iris dataset`_, which is available as a serialized numpy array within the deep learning `data directory`_. There are assumptions made in the implementations of the deep learning code base, especially with respect to the dimensionality of data, but the code is generalized enough to work on other tasks.


Setup
=====

Requirements
------------
- Python 3.2+ (`3.6.6` *suggested*)
- NumPy 1.15 (Earlier versions not tested)

NumPy can be installed via pip: ``pip install numpy``

Environment
...........
All project code has been developed on Linux *(Ubuntu 16.04, 18.04)*, but as long as you have your python environment setup with NumPy and CLI access, it should work on your machine. I would also suggest using a virtualenv manager like pyenv_.

****

Running the model
=================
First, clone this repo:
    ``git clone --depth=1 https://github.com/evdcush/AI-Sandbox.git``
Navigate to the deep_learning folder:
    ``cd sandbox/deep_learning``
Run the model via ``train.py``:
    ``python train.py``

You should see the training and test results for the classifier model:

Something like::

    # Model Summary:

    NeuralNetwork
      Layers:
         1 : Dense (4, 163)
              : Sigmoid
         2 : Dense (163, 3)

    - OPTIMIZER : SGD
    - OBJECTIVE : SoftmaxCrossEntropy

    # Training results, 1500 iterations
    #------------------------------
                Error   |  Accuracy
    * Average: 0.64273  |  0.81241
    *  Median: 0.63921  |  0.83333
    #------------------------------

    # Test results, 30 samples
    #------------------------------
                Error   |  Accuracy
    * Average: 0.47390  |  0.90000
    *  Median: 0.50770  |  1.00000
    #------------------------------






Default model settings are configured as follows:

:Training iterations: 1500
:Batch size: 6
:Channels: [4, 163, 3]
:Activation: Logistic sigmoid
:Optimizer: ``SGD``
:Objective function: Softmax Cross Entropy


Model Options
-------------
The model, as defined on this dataset, can be configured for other settings that can be specified in ``train.py`` or simply passed as arguments through STDIN, for example, the following line:

``python train.py -i 500 -b 12 -o adam -a selu -c 4 29 41 3``

Will train the model for 500 **i**\ terations, with **b**\ atch-size 12, using `selu` **a**\ ctivations, the Adam **o**\ ptimizer, and **c**\ hannels [4, 29, 41, 3].


While the ``SGD`` optimizer can be sensitive to network configuration (notably with channels), ``adam`` is robust and can converge with almost any network config.


|

There are many different settings that can be specified through the CLI, and you can review them all in ``utils.Parser``.

Training options quick-reference
................................

-i int, --num_iters  Number of training iterations
-b int, --batch_size  Training mini-batch sizes.

    This defines how many samples are passed to the model in one training iteration.

-a ACTIVATION, --activation  Activation function used in the network.

    Available activations: ``relu, elu, selu, softplus, sigmoid, tanh, swish, softmax``

-o OPTIMIZER, --optimizer  Model optimizer.

    Available optimizers: ``sgd, adam``



****

Known issues
============
None...yet. Please let me know if you have any issues with the code!

The model performs as expected on the Iris dataset, but there are some intra-module inconsistencies, missing features, and cleanup required.

The most notable lacking feature currently is the inability to serialize or save the model parameters. A lot of that plumbing is in place, such as how parameters are stored and accessed in layers, and the model pathing and constants in utils, but it has not been implemented yet.


.. Substitutions:

.. PROJECT FILES:
.. _deep_learning directory: sandbox/deep_learning
.. |deep_learning directory| replace:: deep_learning directory
.. _functions.py: sandbox/deep_learning/functions.py
.. _layers.py: sandbox/deep_learning/layers.py
.. _network.py: sandbox/deep_learning/network.py
.. _initializers.py: sandbox/deep_learning/initializers.py
.. _optimizers.py: sandbox/deep_learning/optimizers.py
.. _utils.py: sandbox/deep_learning/utils.py
.. _train.py: sandbox/deep_learning/train.py

.. LOCAL FILES:
.. _BSD-3-Clause-Clear: LICENSE
.. _Iris dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set

.. |Iris dataset| replace:: Iris dataset
.. _data directory: sandbox/data/Iris

.. OTHER:
.. _pyenv: https://github.com/pyenv/pyenv
.. |pyenv| replace:: pyenv
