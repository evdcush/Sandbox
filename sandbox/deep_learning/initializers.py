""" Initializers for model parameters

There are 3 types of initializations available:

# Normal
#----------
# Array values sampled from a scaled random normal distribution
Normal Initializers:
  * HeNormal : gaussian scaled by input channels
  * GlorotNormal : gaussian scaled by input and output channels

# Uniform
#----------
# Array values sampled from a random uniform distribution
Uniform Initializers:
  * GlorotUniform : values drawn from within an interval determined
                    by input and output channels


# Constant
#----------
# Array values initialized to some constant
Note: Constant can accept any fill value, but the following initializers
      are provided for convenience and frequency of use
Constant Initializers:
  * Zeros : array of all zeros
  * Ones : array of all ones
"""

import numpy as np

#==============================================================================
#------------------------------------------------------------------------------
#                         Standard Initializers
#------------------------------------------------------------------------------
#==============================================================================

#==============================================================================
# Base initialization functions:
#  Initializer, Normal, Uniform, Constant
#==============================================================================

# Initializer
# -----------
# inherits :
# derives : Normal, Uniform, Constant
class Initializer:
    """ Base class for initializer """
    dtype_ = np.float32
    def __init__(self, **kwargs):
        for attribute, value in kwargs.items():
            setattr(self, attribute, value)

    @classmethod
    def get_class_name(cls):
        return cls.__name__

    def __call__(self, kdims):
        """ initialize an array with values drawn from
        a distribution
        """
        raise NotImplementedError

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Normal
# ------
# inherits : Initializer
# derives : HeNormal, GlorotNormal
class Normal(Initializer):
    """ Initializes an array with a normal distribution

    Values are drawn from a Gaussian distribution with
    mean 0 and stdv 'scale'
    """
    def __init__(self, scale=0.05, **kwargs):
        super().__init__(**kwargs)
        self.normal = np.random.normal
        self.scale = 0.05

    def __call__(self, kdims):
        scale = self.scale
        size = kdims
        param_array = self.normal(scale=scale, size=size).astype(self.dtype_)
        return param_array

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Uniform
# -------
# inherits : Initializer
# derives : GlorotUniform
class Uniform(Initializer):
    """ Initializes an array with a uniform distribution

    Values are drawn uniformly from the interval
    [-scale, scale)
    """
    def __init__(self, scale=0.05, **kwargs):
        super().__init__(**kwargs)
        self.uniform = np.random.uniform
        self.scale = 0.05

    def __call__(self, kdims):
        scale = self.scale
        low, high = -scale, scale
        dtype = self.dtype_
        size = kdims
        param_array = self.uniform(low=low, high=high, size=kdims).astype(dtype)
        return param_array

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Constant
# --------
# inherits : Initializer
# derives  : Zeros
class Constant(Initializer):
    """ Initializes array with repeated constants """
    def __init__(self, fill_value=1.0):
        self.fill_value = fill_value

    def __call__(self, kdims):
        param_array = np.full(kdims, self.fill_value, dtype=self.dtype_)
        return param_array

#==============================================================================
# Initializers
#==============================================================================

#------------------------------------------------------------------------------
# Normal :
#  HeNormal, GlorotNormal
#------------------------------------------------------------------------------

class HeNormal(Normal):
    """ Inititalizes array from scaled Gaussian

    Values drawn independently from a Gaussian distrib.
    with mean 0, and stddev:
      scale * sqrt(2/fan_in)

    Where 'fan_in' = number of input units (input channels)

    """
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(scale, **kwargs)

    def __call__(self, kdims):
        scale = self.scale
        fan_in = kdims[0]
        stdv = scale * np.sqrt(2 / fan_in)
        param_array = self.normal(scale=stdv, size=kdims).astype(self.dtype_)
        return param_array


class GlorotNormal(Normal):
    """ Inititalizes array from scaled Gaussian

    Values drawn independently from a Gaussian distrib.
    with mean 0, and stddev:
      scale * sqrt(2/(fan_in + fan_out))

    Where
    'fan_in'  = number of input units (input channels)
    'fan_out' = number of output units (output channels)

    """
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(scale, **kwargs)

    def __call__(self, kdims):
        scale = self.scale
        fan = sum(kdims)
        stdv = scale * np.sqrt(2 / fan)
        param_array = self.normal(scale=stdv, size=kdims).astype(self.dtype_)
        return param_array


#------------------------------------------------------------------------------
# Uniform :
#  GlorotUniform
#------------------------------------------------------------------------------

class GlorotUniform(Uniform):
    """ Initializes an array with a uniform distribution

    Values are drawn uniformly from the interval
    [-m, m)

    Where
    m = scale * sqrt(6/(fan_in + fan_out))
    'fan_in'  = number of input units (input channels)
    'fan_out' = number of output units (output channels)

    """
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(scale, **kwargs)

    def __call__(self, kdims):
        scale = self.scale
        fan = sum(kdims)
        m = scale * np.sqrt(6 / fan)
        param_array = self.uniform(-m, m, size=kdims).astype(self.dtype_)
        return param_array

#------------------------------------------------------------------------------
# Constant :
#  Zeros, Ones
#------------------------------------------------------------------------------

class Zeros(Constant):
    """ Initializes array with all zeros """
    def __init__(self, fill_value=0.0):
        super().__init__(fill_value)

class Ones(Constant):
    """ Initializes array with all ones """
    def __init__(self, fill_value=1.0):
        super().__init__(fill_value)


#==============================================================================
#------------------------------------------------------------------------------
#                    Pretrained variable Initializers
#------------------------------------------------------------------------------
#==============================================================================

''' # TODO
class Pretrained:
    """ Restores a pretrained variable """
    def __init__(self, trained_params):
        self.trained_params = trained_params


    def __call__(self, kdims):
        pass
'''
