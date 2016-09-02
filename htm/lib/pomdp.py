# encoding: utf-8

import os
import json
import subprocess
from distutils import spawn

import numpy as np

from .py23 import TemporaryDirectory


SOLVER_NAME = 'pomdp-solve'


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
DECIMALS = 5
NUMBER_FORMAT = '{:0.' + str(DECIMALS) + 'f}'


def _as_list(lst_or_int):
    if isinstance(lst_or_int, int):
        return list(range(lst_or_int))
    else:
        return lst_or_int


def _dump_list(lst):
    return ' '.join([str(x) for x in lst])


def _dump_list_or_count(lst_or_int):
    if isinstance(lst_or_int, int):
        return str(lst_or_int)
    else:
        return _dump_list(lst_or_int)


def _dump_1d_array(a):
    # Make sure that sum stays the same even after trunc
    trunc_sum = np.around(a.sum(), decimals=DECIMALS)
    trunc = np.around(a, decimals=DECIMALS)
    imax = np.argmax(trunc)
    # Compensate on max to avoid negative values
    trunc[imax] += trunc_sum - trunc.sum()
    return ' '.join([NUMBER_FORMAT.format(x) for x in trunc])


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

    """Partially observable Markov model.

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
    :solver_path: string
        Path in which to look for the executable (default to $PATH)
    """

    def __init__(self, T, O, R, start, discount, states=None, actions=None,
                 observations=None, values='reward', solver_path=None):
        # Defaults for actions, states and observations
        a, s, o = O.shape
        self._init_states(states, s)
        self._init_actions(actions, a)
        self._init_observations(observations, o)
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
        self._assert_shapes()
        self._assert_normal()
        self._assert_unique()
        if discount > 1 or discount < 0:
            raise ValueError('Discount factor must be ≤ 1 and ≥ 0.')
        self.discount = discount
        self._solver_path = spawn.find_executable(SOLVER_NAME,
                                                  path=solver_path)
        if self._solver_path is None:
            raise ImportError('Could not find executable for pomdp-solve.')

    def _init_states(self, states, s):
        if states is not None:
            self._s = list(states)
        else:
            self._s = s

    def _init_actions(self, actions, a):
        if actions is not None:
            self._a = list(actions)
        else:
            self._a = a

    def _init_observations(self, observations, o):
        if observations is not None:
            self._o = list(observations)
        else:
            self._o = o

    @property
    def states(self):
        return _as_list(self._s)

    @property
    def actions(self):
        return _as_list(self._a)

    @property
    def observations(self):
        return _as_list(self._o)

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

    def _assert_unique(self):
        message = "Found duplicate {}: {}"

        def assert_no_dup(lst, name):
            if not len(set(lst)) == len(lst):
                dup = list(lst)
                for a in set(lst):
                    dup.remove(a)
                raise ValueError(message.format(name, dup))

        assert_no_dup(self.states, 'states(s)')
        assert_no_dup(self.actions, 'action(s)')
        assert_no_dup(self.observations, 'observation(s)')

    def belief_update(self, a, o, b):
        new_b = b.dot(self.T[a, ...]) * self.O[a, :, o]
        return new_b / new_b.sum()

    def dump(self):
        """Write POMDP description following:
        `<http://www.pomdp.org/code/pomdp-file-spec.html>`_
        """
        preamble = PREAMBLE_FMT.format(
            discount=self.discount,
            states=_dump_list_or_count(self._s),
            actions=_dump_list_or_count(self._a),
            observations=_dump_list_or_count(self._o))
        start = "start: {}".format(_dump_1d_array(np.asarray(self.start)))
        T = _dump_3d_array(self.T, 'T', self.actions)
        O = _dump_3d_array(self.O, 'O', self.actions)
        R = _dump_4d_array(self.R, 'R', self.actions, self.states)
        return '\n\n'.join([preamble, start, T, O, R])

    def dump_to(self, path, name):
        full_path = os.path.join(path, name + '.pomdp')
        with open(full_path, 'w') as f:
            f.write(self.dump())
        return full_path

    def solve(self, timeout=None, n_iterations=None, method='incprune',
              grid_type=None):
        """
        :param method: incprune | grid (incprune)
        :param grid_type: simplex | pairwise (simplex)
        """
        name = 'tosolve'
        args = []
        if timeout is not None:
            args.extend(['-time_limit', str(timeout)])
        if n_iterations is not None:
            args.extend(['-horizon', str(n_iterations)])
        if method == 'grid':
            if grid_type is None:
                grid_type = 'simplex'
            args.extend(['-method', method, '-fg_type', grid_type])
        with TemporaryDirectory() as tmpdir:
            pomdp_file = self.dump_to(tmpdir, name)
            args.extend(['-o', name, '-pomdp', pomdp_file])
            subprocess.check_call(
                [self._solver_path] + args, cwd=tmpdir,
                stdout=subprocess.PIPE)
            # TODO: add timeout=timeout for python 3
            # Note: actually timeout + 1 should be used to give a margin for
            # pomdp-solve for parsing and cleaning.
            return self.load_policy_from(tmpdir, name)

    def load_policy_from(self, path, name):
        value_function_file = os.path.join(path, name + '.alpha')
        policy_graph_file = os.path.join(path, name + '.pg')
        with open(value_function_file, 'r') as vf:
            actions, vf = parse_value_function(vf)
        with open(policy_graph_file, 'r') as pf:
            actions2, pg = parse_policy_graph(pf)
        assert(actions == actions2)
        assert(max(actions) < len(self.actions))
        assert(np.max(pg) <= len(pg))
        action_names = [self.actions[a] for a in actions]
        return GraphPolicy(action_names, self.observations, pg, vf, self.start)


class GraphPolicy:

    def __init__(self, actions, observations, transitions, values, start):
        self.actions = actions
        self.observations = observations
        self.transitions = np.asarray(transitions)
        assert(self.transitions.shape == (self.n_nodes, len(observations)))
        self.values = values
        self.init = self.get_node_from_belief(start)

    @property
    def n_nodes(self):
        return len(self.actions)

    def get_node_from_belief(self, b):
        return self.values.dot(b[:, np.newaxis]).argmax()

    def get_action(self, current):
        return self.actions[current]

    def next(self, current, observation):
        return self.transitions[current, self.observations.index(observation)]

    def to_json(self):
        return json.dumps({'actions': self.actions,
                           'observations': self.observations,
                           'transitions': self.transitions.tolist(),
                           'values': self.values.tolist(),
                           })


class GraphPolicyRunner(object):

    def __init__(self, graph_policy):
        self.gp = graph_policy
        self.reset()

    def reset(self, belief=None):
        if belief is not None:
            self.current = self.gp.get_node_from_belief(belief)
        else:
            self.current = self.gp.init

    def get_action(self):
        return self.gp.get_action(self.current)

    def step(self, observation):
        self.current = self.gp.next(self.current, observation)
        if self.current is None:
            raise ValueError('Got unexpected observation')


class GraphPolicyBeliefRunner(GraphPolicyRunner):

    def __init__(self, graph_policy, pomdp):
        self.gp = graph_policy
        self.pomdp = pomdp
        self.reset()

    def reset(self, belief=None):
        if belief is None:
            belief = self.pomdp.start
        self.current_belief = belief
        super(GraphPolicyBeliefRunner, self).reset(belief=belief)

    def step(self, observation):
        a = self.pomdp.actions.index(self.get_action())
        o = self.pomdp.observations.index(observation)
        b = self.pomdp.belief_update(a, o, self.current_belief)
        self.reset(belief=b)
