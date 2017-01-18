import numpy as np
from itertools import chain
from htm.task import (AbstractAction, SequentialCombination,
                      AlternativeCombination, LeafCombination,
                      ParallelCombination)


def unique(l):
    return list(set(l))


class _HTMToDAG:

    def __init__(self, node):
        self.nodes = []
        self.succs = []
        self.init, _ = self.get_initial_final(node)

    def push(self, node):
        self.nodes.append(node)
        self.succs.append([])
        return len(self.nodes) - 1

    def get_initial_final(self, node):
        """Visits the graph, recording the leaves and returns list of indices
        of initial and final leaves in the node.
        """
        if isinstance(node, LeafCombination):
            i = self.push(node)
            first, last = [i], [i]
        else:
            if isinstance(node, ParallelCombination):
                node = node.to_alternative()
            first, last = zip(*[self.get_initial_final(c) for c in node.children])
            if isinstance(node, SequentialCombination):
                for ff, ll in zip(first[1:], last[:-1]):
                    for l in ll:
                        self.succs[l] = ff
                first = first[0]
                last = last[-1]
            elif isinstance(node, AlternativeCombination):
                first = unique(chain(*first))
                last = unique(chain(*last))
        return (first, last)


CONSUMES = 0
USES = 1
CONSUMES_SOME = 2


class SupportedAction(AbstractAction):

    pass


class BringTop(SupportedAction):

    def __init__(self):
        super(BringTop, self).__init__('Bring Top')
        self.conditions = [(CONSUMES, 'top')]


class AssembleFoot(SupportedAction):

    def __init__(self, leg):
        super(AssembleFoot, self).__init__('Assemble foot on ' + leg)
        self.conditions = [(CONSUMES, 'foot'),
                           (CONSUMES, 'leg'),
                           (USES, 'screwdriver'),
                           (CONSUMES_SOME, 'screws'),
                           ]


class AssembleTopJoint(SupportedAction):

    def __init__(self, leg):
        super(AssembleTopJoint, self).__init__('Assemble joint on ' + leg)
        self.conditions = [(CONSUMES, 'joint'),
                           (USES, 'screwdriver'),
                           (CONSUMES_SOME, 'screws'),
                           ]


class AssembleLegToTop(SupportedAction):

    def __init__(self, leg):
        super(AssembleLegToTop, self).__init__('Assemble {} to top'.format(leg))
        self.conditions = [(CONSUMES, 'joint'),
                           (USES, 'screwdriver'),
                           (CONSUMES_SOME, 'screws'),
                           ]


class SupportivePOMDP:
    """
    Each action has a condition attribute that is a pair (condition, object)
    where objects are represented as strings and condition is one of:
    (CONSUMES, USES, CONSUMES_SOME).

    The HTM feature has the following values:
    - one for each node in the DAG representation of the HTM
    - one final state
    """

    A_WAIT = 0
    A_HOLD = 1
    O_NONE = 0
    F_HTM = 0

    p_consume_all = .5
    r_subtask = 10.
    r_final = 100.
    intrinsic_cost = 1

    preferences = ['hold-preference']
    p_preferences = [0.5]

    states = None
    observations = ['None']
    n_observations = 1

    def __init__(self, htm):
        h2d = _HTMToDAG(htm)
        self.htm_nodes = h2d.nodes
        assert(len(self.htm_nodes) < 128)  # we use dtype=np.int8
        self.htm_succs = h2d.succs
        self.htm_init = h2d.init
        self._populate_conditions()
        self._skip_to_obj = 1 + len(self.preferences) + 1
        self.n_features = self._skip_to_obj + len(self.objects)
        self.n_states = self.n_htm_states * (2 ** self.n_features)
        self._skip_to_a_obj = 2
        self.n_actions = self._skip_to_a_obj + 2 * len(self.objects)

    def get_object_id(self, object_name):
        # Note: this is not efficient
        if object_name not in self.objects:
            self.objects.append(object_name)
        return self.objects.index(object_name)

    def _populate_conditions(self):
        self.objects = []
        self.htm_conditions = [[] for _ in self.htm_nodes]
        for n, conditions in zip(self.htm_nodes, self.htm_conditions):
            for c, o in n.action.conditions:
                conditions.append((c, self.get_object_id(o)))

    @property
    def n_htm_states(self):
        """Number of nodes in the DAG plus 1 for the final state.
        """
        return len(self.htm_nodes) + 1

    @property
    def features(self):
        return ['HTM'] + self.preferences + ['holding'] + self.objects

    @property
    def actions(self):
        return ['wait', 'hold'] + list(chain(*[
            ['bring ' + o, 'remove ' + o] for o in self.objects]))

    def belief_update(self, a, o, b):
        raise NotImplemented

    def _update_for_transition(self, s, node):
        """Computes reward and modifies state to match transition from node
        to a random successor.
        """
        s[self.F_HTM] = np.choice(self.htm_succs[node])
        return sum([self._update_for_condition(s, c, o)
                    for c, o in self.htm_conditions[node]])

    def _obj_feat(self, obj):
        return self._skip_to_obj + obj

    def _obj_from_action(self, a):
        return (a - self._skip_to_a_obj) // 2

    def _is_bring(self, a):
        return (a - self._skip_to_a_obj) % 2 == 0

    def _bring(self, obj):
        return self._skip_to_a_obj + 2 * obj

    def _remove(self, obj):
        return self._bring(obj) + 1

    def _update_for_condition(self, s, c, o):
        """Computes reward and modifies state according to conditions.
        """
        f_o = self._obj_feat(o)
        r = self._cost_get(o) if s[f_o] else 0.  # all conditions need object to be there
        if c == CONSUMES:
            s[f_o] = 0
        elif c == CONSUMES_SOME:
            s[f_o] = 0 if np.random.random() < self.p_consume_all else 1.
        return r

    def _cost_get(self, o):
        return -10  # Cost for the human to get the object

    def sample_transition(self, a, s):
        new_s = s.copy()
        if a == self.A_WAIT or a == self.A_HOLD:
            r = 0 if a == self.A_WAIT else -self.intrinsic_cost
            if s[self.F_HTM] == self.n_htm_states:  # Final state
                obs = self.O_NONE
            else:
                r += self._update_for_transition(new_s, s[self.F_HTM])
                obs = self.O_NONE
                if new_s[self.F_HTM] == self.n_htm_states:
                    r += self.r_final
                else:
                    r += self.r_subtask
        else:
            f_o = self._obj_feat(self._obj_from_action(a))
            if self._is_bring(a):  # Bring
                new_s[f_o] = 1
                obs = self.O_NONE
            else:  # Remove
                new_s[f_o] = 0
                obs = self.O_NONE
            # TODO: add random transitions on other features
            r = -self.intrinsic_cost  # Intrinsic action cost
        return new_s, obs, r

    def sample_start(self):
        htm_id = np.random.choice(self.htm_init)
        s = np.zeros((self.n_features), np.int8)
        s[self.F_HTM] = htm_id
        for i, p in enumerate(self.preferences):
            s[1 + i] = 1 if np.random.random() < self.p_preferences[i] else 0
        return s

    @property
    def start(self):
        raise NotImplementedError
