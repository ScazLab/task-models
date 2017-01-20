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


class _SupportivePOMDPState:
    """Enables conversion of states between integers and array representation
    """

    def __init__(self, n_htm_states, n_preferences, n_body_features, n_objects, s=0):
        self._shift_body = n_objects
        self._shift_pref = self._shift_body + n_body_features
        self._shift_htm = self._shift_pref + n_preferences
        self.s = s

    def __str__(self):
        return "<{}: {} {} {} {}>".format(
            self.s,
            self.htm,
            "".join([str(self.has_preference(i))
                      for i in range(self._shift_htm - self._shift_pref)]),
            "".join([str(self.has_body_feature(i))
                      for i in range(self._shift_pref - self._shift_body)]),
            "".join([str(self.has_object(i))
                      for i in range(self._shift_body)]))

    @property
    def htm(self):
        return self.s >> self._shift_htm

    @htm.setter
    def htm(self, n):
        self.s += (n << self._shift_htm) - (self.htm << self._shift_htm)

    def _get_bit(self, i):
        return (self.s >> i) % 2

    def _set_bit(self, i, b):
        self.s += (b << i) - (self._get_bit(i) << i)

    def has_preference(self, i):
        return self._get_bit(self._shift_pref + i)

    def set_preference(self, i, pref):
        self._set_bit(self._shift_pref + i, pref)

    def has_body_feature(self, i):
        return self._get_bit(self._shift_body + i)

    def set_body_feature(self, i, b):
        self._set_bit(self._shift_body + i, b)

    has_object = _get_bit
    set_object = _set_bit

    def to_int(self):
        return self.s


class SupportivePOMDP:
    """
    Each action has a condition attribute that is a pair (condition, object)
    where objects are represented as strings and condition is one of:
    (CONSUMES, USES, CONSUMES_SOME).

    The HTM feature has the following values:
    - one for each node in the DAG representation of the HTM
    - one final state

    Note on state representation:
    - for public API, states are integers, denoted as `s`
    - for private methods states may be using the _SupportivePOMDPState;
    they are then denoted as `_s`.
    """

    A_WAIT = 0
    A_HOLD = 1
    O_NONE = 0

    p_consume_all = .5
    r_subtask = 10.
    r_final = 100.
    intrinsic_cost = 1

    preferences = ['hold-preference']
    p_preferences = [0.5]

    states = None
    observations = ['None']
    n_observations = 1

    def __init__(self, htm, discount=1.):
        self.discount = discount
        h2d = _HTMToDAG(htm)
        self.htm_nodes = h2d.nodes
        assert(len(self.htm_nodes) < 128)  # we use dtype=np.int8
        # set final state as successors of last actions in HTM
        self.htm_succs = [[self.htm_final] if len(s) == 0 else s for s in h2d.succs]
        self.htm_init = h2d.init
        self._populate_conditions()
        self.n_states = self.n_htm_states * (
                2 ** (1 + len(self.preferences) + 1 + len(self.objects)))
        self._skip_to_a_obj = 2
        self.n_actions = self._skip_to_a_obj + 2 * len(self.objects)

    def _int_to_state(self, s=0):
        return _SupportivePOMDPState(self.n_htm_states, len(self.preferences),
                                     1, len(self.objects), s=s)

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
    def htm_final(self):
        return len(self.htm_nodes)

    @property
    def features(self):
        return ['HTM'] + self.preferences + ['holding'] + self.objects

    @property
    def states(self):
        return [str(self._int_to_state(i)) for i in range(self.n_states)]

    @property
    def actions(self):
        return ['wait', 'hold'] + list(chain(*[
            ['bring ' + o, 'remove ' + o] for o in self.objects]))

    def belief_update(self, a, o, b):
        raise NotImplemented

    def _update_for_transition(self, _s, node):
        """Computes reward and modifies state to match transition from node
        to a random successor.
        """
        _s.htm = np.random.choice(self.htm_succs[node])
        return sum([self._update_for_condition(_s, c, o)
                    for c, o in self.htm_conditions[node]])

    def _obj_from_action(self, a):
        return (a - self._skip_to_a_obj) // 2

    def _is_bring(self, a):
        return (a - self._skip_to_a_obj) % 2 == 0

    def _bring(self, obj):
        return self._skip_to_a_obj + 2 * obj

    def _remove(self, obj):
        return self._bring(obj) + 1

    def _update_for_condition(self, _s, c, obj):
        """Computes reward and modifies state according to conditions.
        """
        r = self._cost_get(obj) if _s.has_object(obj) else 0.
        # note: all conditions need object to be there
        if c == CONSUMES:
            _s.set_object(obj, 0)
        elif c == CONSUMES_SOME:
            _s.set_object(obj, 0 if np.random.random() < self.p_consume_all else 1)
        return r

    def _cost_get(self, o):
        return -10  # Cost for the human to get the object

    def sample_transition(self, a, s):
        _s = self._int_to_state(s)
        _new_s = self._int_to_state(s)
        if a == self.A_WAIT or a == self.A_HOLD:
            r = 0 if a == self.A_WAIT else -self.intrinsic_cost
            if _s.htm == self.htm_final:  # Final state
                obs = self.O_NONE
            else:
                r += self._update_for_transition(_new_s, _s.htm)
                obs = self.O_NONE
                if _new_s.htm == self.htm_final:
                    r += self.r_final
                else:
                    r += self.r_subtask
        else:
            obj = self._obj_from_action(a)
            if self._is_bring(a):  # Bring
                _new_s.set_object(obj, 1)
                obs = self.O_NONE
            else:  # Remove
                _new_s.set_object(obj, 0)
                obs = self.O_NONE
            # TODO: add random transitions on other features
            r = -self.intrinsic_cost  # Intrinsic action cost
        return _new_s.to_int(), obs, r

    def sample_start(self):
        htm_id = np.random.choice(self.htm_init)
        _s = self._int_to_state()
        _s.htm = htm_id
        for i, p in enumerate(self.p_preferences):
            _s.set_preference(i, 1 if np.random.random() < p else 0)
        return _s.to_int()

    @property
    def start(self):
        raise NotImplementedError
