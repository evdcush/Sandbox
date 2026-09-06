""" Training script for iris model
"""
import numpy as np

import utils
from utils import SessionStatus, classification_accuracy
from network import NeuralNetwork


# args parser
#------------------
# config contains all model and session setup options
parser = utils.Parser()
config = parser.parse_args()
parser.print_args()

# Load data
#------------------
iris_data = utils.IrisDataset()
num_test_samples = iris_data.X_test.shape[0]


#==============================================================================
# Model, session config
#==============================================================================
# Model config
#------------------
channels   = config.channels
activation = config.activation
optimizer  = config.optimizer
objective  = config.objective
dropout    = config.dropout
learning_rate = config.learn_rate # 0.01, omitted, opt defaults well configured

# Session config
#------------------
num_iters  = config.num_iters
batch_size = config.batch_size
verbose = config.verbose


# Model initialization
#------------------------------------------------------------------------------

# Instantiate model
#------------------
np.random.seed(config.seed)
model = NeuralNetwork(channels, activation=activation, use_dropout=dropout)
opt   = optimizer()
objective = objective()

# Model status reporter
#------------------
sess_status = SessionStatus(model, opt, objective, num_iters, num_test_samples)


#==============================================================================
# Train
#==============================================================================
np.random.seed(utils.RNG_SEED_DATA)

for step in range(num_iters):
    # batch data
    #------------------
    x, y = iris_data.get_batch(step, batch_size)

    # forward pass
    #------------------
    y_hat = model.forward(x)
    error, class_scores = objective(y_hat, y)
    accuracy = classification_accuracy(class_scores, y)
    sess_status(step, error, accuracy, show=verbose)

    # backprop and update
    #------------------
    grad_loss = objective(backprop=True)
    model.backward(grad_loss)
    model.update(opt)


# Finished training
#------------------------------------------------------------------------------


#==============================================================================
# Validation
#==============================================================================

# Test
#------------------
for i in range(num_test_samples):
    x, y = iris_data.get_batch(i, test=True)

    # forward pass
    #------------------
    y_hat = model.forward(x, test=True)
    error, class_scores = objective(y_hat, y)
    accuracy = utils.classification_accuracy(class_scores, y)
    sess_status(i, error, accuracy, show=verbose, test=True)


# Finished test
#------------------------------------------------------------------------------
#==============================================================================

# Model performance summary
sess_status.summarize_model(True, True)
