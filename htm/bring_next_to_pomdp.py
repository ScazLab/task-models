import numpy as np

from .lib.pomdp import POMDP
from .task import (AbstractAction, SequentialCombination,
                   AlternativeCombination, LeafCombination,
                   ParallelCombination)
from .task_to_pomdp import _name_radix


class CollaborativeAction(AbstractAction):

    def __init__(self, name, obj):
        super(CollaborativeAction, self).__init__(name=name)
        self.obj = obj


class _NodeToPOMDP(object):

    observations = ['none', 'yes', 'no', 'error']

    o_none = [1., 0., 0., 0.]
    o_yes = [0., 1., 0., 0.]
    o_no = [0., 0., 1., 0.]
    o_err = [0., 0., 0., 1.]
    o_maybe_good = [.5, 0., 0., .5]

    init = None     # index of init states
    start = None    # start probabilities
    states = None   # state names

    def update_T(self, T, builder, s_start, s_next, s_next_probas):
        """:param builder: POMDP builder
        """
        raise NotImplementedError

    def update_O(self, O, builder, s_start, s_next):
        raise NotImplementedError

    def update_R(self, R, builder, s_start, s_next):
        raise NotImplementedError

    @staticmethod
    def from_node(node):
        if isinstance(node, LeafCombination):
            return _LeafToPOMDP(node)
        elif isinstance(node, SequentialCombination):
            return _SequenceToPOMDP(node)
        elif isinstance(node, AlternativeCombination):
            return _AlternativesToPOMDP(node)
        elif isinstance(node, ParallelCombination):
            return _AlternativesToPOMDP(node.to_alternative())
        else:
            raise ValueError('Unkown combination: ' + type(node))


class _LeafToPOMDP(_NodeToPOMDP):

    init = [0]  # only state
    start = [1.]

    act = 'get'
    com = 'ask'

    def __init__(self, leaf):
        self.leaf = leaf
        radix = _name_radix(leaf.action)  # action name
        obj = self.leaf.action.obj  # object name
        self.states = ['before-' + radix]
        self.get = self.act + '-' + obj
        self.ask = self.com + '-' + obj

    def update_T(self, T, builder, s_start, s_next, s_next_probas):
        T[:, s_start, s_start] = 1.
        get = builder.action_indices[self.get]
        T[get, s_start, s_start] = 0.
        T[get, s_start, s_next] = s_next_probas

    def update_O(self, O, builder, s_start, s_next):
        get = builder.action_indices[self.get]
        ask = builder.action_indices[self.ask]
        O[get, s_next, :] = self.o_maybe_good
        O[ask, s_start, :] = self.o_yes

    def update_R(self, R, builder, s_start, s_next):
        get = builder.action_indices[self.get]
        R[get, s_start, s_next, :] = builder.cost_get


class HTMToPOMDP:

    end = -1

    def __init__(self, t_com, t_get, t_err, objects, end_reward=10.):
        self.cost_com = t_com
        self.cost_get = t_get
        self.cost_err = t_err
        self.end_reward = end_reward
        self.get = ['get-' + o for o in objects]
        self.ask = ['ask-' + o for o in objects]
        self.actions = self.get + self.ask
        self.get_indices = [i for i in range(len(objects))]
        self.ask_indices = [len(objects) + i for i in range(len(objects))]
        self.action_indices = {a: i for i, a in enumerate(self.actions)}

    def update_T_end(self, T, init):
        # Loop on success
        T[:, self.end, init] = 1. / len(init)

    def init_O(self, O):
        O[self.get_indices, :, :] = _NodeToPOMDP.o_err
        O[self.ask_indices, :, :] = _NodeToPOMDP.o_no

    def init_R(self, R):
        R[self.get_indices, ...] = self.cost_err
        R[self.ask_indices, ...] = self.cost_com

    def update_R_end(self, R):
        R[:, self.end, ...] = -self.end_reward   # Get reward on end

    def task_to_pomdp(self, task):
        n2p = _NodeToPOMDP.from_node(task.root)
        states = [s for s in n2p.states]
        states.append('end')
        start = np.zeros(len(states))
        start[n2p.init] = n2p.start
        n_s = len(states)
        n_a = len(self.actions)
        n_o = len(n2p.observations)
        end = n_s - 1
        T = np.zeros((n_a, n_s, n_s))
        n2p.update_T(T, self, 0, [end], [1.])
        self.update_T_end(T, n2p.init)
        O = np.zeros((n_a, n_s, n_o))
        self.init_O(O)
        n2p.update_O(O, self, 0, [end])
        R = np.zeros((n_a, n_s, n_s, n_o))
        self.init_R(R)
        n2p.update_R(R, self, 0, [end])
        self.update_R_end(R)
        return POMDP(T, O, R, start, discount=1., states=states,
                     actions=self.actions, observations=n2p.observations,
                     values='cost')
