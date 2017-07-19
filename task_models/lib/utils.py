from json import JSONEncoder

import numpy as np


def assert_normal(array, name='array'):
    message = "Probabilities in {} should sum to 1."
    if not np.allclose(array.sum(-1), 1.):
        raise ValueError(message.format(name))


class NPEncoder(JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NPEncoder, self).default(obj)
