"""
Collection of commonly used functions, structures, or configs
"""
import os
import sys
import code
import argparse
import numpy as np
import dataset


#-----------------------------------------------------------------------------#
#                                 DATA UTILS                                  #
#-----------------------------------------------------------------------------#

# Make DATASETS available in this namespace
DATASETS = dataset.DATASETS

def to_one_hot(Y, num_classes):
    """ make one-hot encoding for truth labels

    Encodes a 1D vector of 0-indexed integer class labels
    into a sparse binary 2D array where every sample has
    length num_classes, with a 1 at the index of its
    constituent label and zeros elsewhere.

    Usage::
        >>> to_one_hot([2, 1, 1, 0], 4)
        array([[0,0,1,0],
               [0,1,0,0],
               [0,1,0,0],
               [1,0,0,0]])

    """
    # Dimensional integrity check on Y
    # handles both ndim = 0 and ndim > 1 cases
    if Y.ndim != 1:
        Y = np.squeeze(Y)
        assert Y.ndim == 1

    # Check num_classes valid
    assert Y.max() < num_classes and num_classes > 0

    # Make one-hot
    n, d = Y.shape[0], num_classes
    one_hot = np.zeros((n, d))
    one_hot[np.arange(n), Y] = 1
    return one_hot.astype(np.int32)

#-----------------------------------------------------------------------------#
#                                   Parser                                    #
#-----------------------------------------------------------------------------#

# Base parser
# ===========
CLI = argparse.ArgumentParser()

# Subcommands
# ===========
def subcmd(*args, **kwargs):
    """Decorator to define a new subcommand in a sanity-preserving way.

    The function will be stored in the ``func`` variable when the parser
    parses arguments so that it can be called directly like so::
        args = cli.parse_args()
        args.func(args)

    Usage example::
        @subcmd("-d", help="Enable debug mode", action="store_true")
        def foo(args):
            print(args)

    Then on the command line::
        $ python cli.py foo -d
    """
    global subparsers
    if subparsers is None:
        subparsers = cli.add_subparsers(dest='subcmd')
    parent = subparsers
    def decorator(func):
        parser = parent.add_parser(func.__name__, description=func.__doc__)
        #for args, kwargs in parser_args:
        #    parser.add_argument(*args, **kwargs)
        parser.add_argument(*args, **kwargs)
        parser.set_defaults(func=func)
    return decorator
