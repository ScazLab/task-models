# encoding: utf-8

from __future__ import print_function

import io
import os
import stat
import subprocess
from pkg_resources import resource_string

import numpy as np

from .py23 import TemporaryDirectory


# This is the content of the binary
SOLVER_NAME = 'pomdp-solve'
POMDP_SOLVE = resource_string(__name__, 'bundle/' + SOLVER_NAME)


def copy_solver(dest):
    path = os.path.join(dest, SOLVER_NAME)
    with io.open(path, 'wb') as f:
        f.write(POMDP_SOLVE)
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)
    return path


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
    return actions, np.vstack(vectors)


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
        self._assert_normal()
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

    def _assert_normal(self):
        message = "Probabilities in {} should sum to 1."

        def assert_normal(array, name):
            if not np.allclose(array.sum(-1), 1.):
                raise ValueError(message.format(name))

        assert_normal(self.start, 'start')
        assert_normal(self.T, 'T')
        assert_normal(self.O, 'O')

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

    def dump_to(self, path, name):
        full_path = os.path.join(path, name + '.pomdp')
        with open(full_path, 'w') as f:
            f.write(self.dump())
        return full_path

    def solve(self):
        out_fmt = '{name}-{pid}.{ext}'
        name = 'tosolve'
        with TemporaryDirectory() as tmpdir:
            pomdp_file = self.dump_to('/tmp', name)
            pomdp_file = self.dump_to(tmpdir, name)
            solver_path = copy_solver(tmpdir)
            solver = subprocess.Popen([solver_path, '-pomdp', pomdp_file],
                                      stdout=subprocess.PIPE)
            if solver.wait() != 0:
                print(solver.stdout.read().decode())  # TODO improve
                raise RuntimeError('Solver failed.')
            pid = solver.pid
            value_function_file = os.path.join(
                tmpdir, out_fmt.format(name=name, pid=pid, ext='alpha'))
            policy_graph_file = os.path.join(
                tmpdir, out_fmt.format(name=name, pid=pid, ext='pg'))
            with open(value_function_file, 'r') as vf:
                actions, vf = parse_value_function(vf)
            with open(policy_graph_file, 'r') as pf:
                actions2, pg = parse_policy_graph(pf)
            assert(actions == actions2)
            assert(max(actions) < len(self.actions))
            action_names = [self.actions[a] for a in actions]
            init = vf.dot(self.start[:, np.newaxis]).argmax()
            return GraphPolicy(action_names, self.observations, pg, init)


class GraphPolicy:

    def __init__(self, actions, observations, transitions, init):
        self.actions = actions
        self.observations = observations
        self.transitions = np.asarray(transitions)
        assert(self.transitions.shape == (self.n_nodes, len(observations)))
        self.init = init

    @property
    def n_nodes(self):
        return len(self.actions)

    def get_action(self, current):
        return self.actions[current]

    def next(self, current, observation):
        return self.transitions[current, self.observations.index(observation)]

    def print(self):
        print('Actions:', self.actions)
        print('Init:', self.init)
        print('Policy graph:')
        print(self.transitions)


class GraphPolicyRunner:

    def __init__(self, graph_policy):
        self.gp = graph_policy
        self.reset()

    def reset(self):
        self.current = self.gp.init

    def get_action(self):
        return self.gp.get_action(self.current)

    def step(self, observation):
        self.current = self.gp.next(self.current, observation)
