import numpy as np


def assert_normal(array, name='array'):
    message = "Probabilities in {} should sum to 1."
    if not np.allclose(array.sum(-1), 1.):
        raise ValueError(message.format(name))
