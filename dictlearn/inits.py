"""
Initializers
"""

import numpy as np
import theano
from blocks.initialization import NdarrayInitialization

class GlorotUniform(NdarrayInitialization):
    """Initialize parameters from an isotropic Gaussian distribution.

    Parameters
    ----------
    std : float, optional
        The standard deviation of the Gaussian distribution. Defaults to 1.
    mean : float, optional
        The mean of the Gaussian distribution. Defaults to 0

    Notes
    -----
    Be careful: the standard deviation goes first and the mean goes
    second!

    """

    def __init__(self):
        pass

    def generate(self, rng, shape):
        if len(shape) == 1:
            return rng.uniform(size=shape, low=-0.00001, high=0.00001).astype(theano.config.floatX)

        if not len(shape) == 2:
            raise NotImplementedError("GlorotUniform doesnt work for " + str(shape) + " shape")

        fan_in, fan_out = shape[0], shape[1]
        s = np.sqrt(6. / (fan_in + fan_out))
        return rng.uniform(size=shape, low=-s, high=s).astype(theano.config.floatX)

    def __repr__(self):
        return "GlorotUniform"
