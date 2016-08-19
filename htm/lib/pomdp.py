import numpy as np


class ValueFunctionParseError(ValueError):
    pass


def parse_value_function(reader):
    has_action = False
    actions = []
    vectors = []
    for line in reader:
        if not line.isspace():
            if has_action:
                # expect vector
                vectors.append(np.fromstring(line, sep=' '))
                has_action = False
            else:
                # expect action
                actions.append(int(line))
                has_action = True
        # else: skip line
    if has_action:
        raise ValueFunctionParseError('Action defined but no vectors follows.')
    return actions, vectors


def parse_policy_graph(reader):
    actions = []
    transitions = []
    for i, line in enumerate(reader):
        line = line.rstrip('\n').rstrip()
        if not line.isspace():
            # 'N A  Z1 Z2 Z3'.split(' ') -> ['N', 'A', '', 'Z1', 'Z2', 'Z3']
            l = line.split(' ')
            n = int(l[0])  # Node name
            assert(n == i)
            actions.append(int(l[1]))
            transitions.append([None if x == '-' else int(x) for x in l[3:]])
    return actions, transitions


PREAMBLE_FMT = """discount: {discount}
values: reward
states: {states}
actions: {actions}
observations: {observations}
"""


def _dump_list(lst):
    return ' '.join([str(x) for x in lst])


def _dump_1d_array(a):
    return ' '.join(['{:0.5f}'.format(x) for x in a])


def _dump_2d_array(a):
    return '\n'.join([_dump_1d_array(x) for x in a])


def _dump_3d_array(a, name, xs):
    """Dump a 3d array for a POMDP file.

    :param a: the array
    :param name: the name of the array in the file
    :param xs: names of the first dimension
    """
    return '\n'.join([
        "{} : {}\n{}".format(name, x, _dump_2d_array(a[ix, :, :]))
        for ix, x in enumerate(xs)
        ])


def _dump_4d_array(a, name, xs, ys):
    """Dump a 4d array for a POMDP file.

    :param a: the array
    :param name: the name of the array in the file
    :param xs: names of the first dimension
    :param ys: names of the second dimension
    """
    return '\n'.join([
        _dump_3d_array(a[ix, :, :, :], name,
                       ["{} : {}".format(x, y) for y in ys])
        for ix, x in enumerate(xs)
        ])


class POMDP:

    """Partially observabla Markov model.

    :param T: array of shape (n_actions, n_states, n_states)
        Transition probabilities (must sum to 1 on last dimension)
    :param O: array of shape (n_actions, n_states, n_observations)
        Observation probabilities (action, *end state*, observation)
        (must sum to 1 on last dimension)
    :param R: array of shape (n_actions, n_states, n_states, n_observations)
        Rewards or cost (must sum to 1 on last dimension)
    :param start: array os shape (n_states)
        Initial state probabilities
    :param discount: discount factor (int)
    :param states: None | iterable of states
        Default to range(n_states).
    :param actions: None | iterable of actions
        Default to range(n_actions).
    :param observations: None | iterable of observations
        Default to range(n_observations).
    :values: ('reward' | 'cost')
        How to interpret reward coefficients.
    """

    def __init__(self, T, O, R, start, discount, states=None, actions=None,
                 observations=None, values='reward'):
        # Defaults for actions, states and observations
        a, s, o = O.shape
        if states is None:
            states = range(s)
        if actions is None:
            actions = range(a)
        if observations is None:
            observations = range(o)
        self.T = T
        self.O = O
        if values == 'reward':
            self.R = R
        elif values == 'cost':
            self.R = -R
        else:
            raise ValueError(
                "Values must be 'reward' of 'cost. Got '{}'.".format(values))
        self.start = start
        self.states = list(states)
        self.actions = list(actions)
        self.observations = list(observations)
        self._assert_shapes()
        if discount > 1 or discount < 0:
            raise ValueError('Discount factor must be ≤ 1 and ≥ 0.')
        self.discount = discount

    def _assert_shapes(self):
        s = len(self.states)
        a = len(self.actions)
        o = len(self.observations)
        message = "Wrong shape for {}: got {}, expected {}."

        def assert_shape(array, name, shape):
            if array.shape != shape:
                raise ValueError(message.format(name, array.shape, shape))

        assert_shape(self.start, 'start', (s,))
        assert_shape(self.T, 'T', (a, s, s))
        assert_shape(self.O, 'O', (a, s, o))
        assert_shape(self.R, 'R', (a, s, s, o))

    def dump(self):
        """Write POMDP description following:
        `<http://www.pomdp.org/code/pomdp-file-spec.html>`_
        """
        preamble = PREAMBLE_FMT.format(
            discount=self.discount,
            states=_dump_list(self.states),
            actions=_dump_list(self.actions),
            observations=_dump_list(self.observations))
        start = "start: {}".format(_dump_1d_array(list(self.start)))
        T = _dump_3d_array(self.T, 'T', self.actions)
        O = _dump_3d_array(self.O, 'O', self.actions)
        R = _dump_4d_array(self.R, 'R', self.actions, self.states)
        return '\n\n'.join([preamble, start, T, O, R])
