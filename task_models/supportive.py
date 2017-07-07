import numpy as np
from itertools import chain
from .task import (AbstractAction, SequentialCombination,
                   AlternativeCombination, LeafCombination,
                   ParallelCombination)
from .lib.pomcp import Horizon


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

    hold = None

    def __init__(self):
        super(BringTop, self).__init__('Bring Top')
        self.conditions = [(CONSUMES, 'top')]


class AssembleLeg(SupportedAction):

    hold = 'h'

    def __init__(self, leg):
        super(AssembleLeg, self).__init__('Assemble ' + leg)
        self.conditions = [(USES, 'joints'),
                           (CONSUMES, 'leg'),
                           (USES, 'screwdriver'),
                           (USES, 'screws'),
                           ]


class AssembleFoot(SupportedAction):

    hold = 'h'

    def __init__(self, leg):
        super(AssembleFoot, self).__init__('Assemble foot on ' + leg)
        self.conditions = [(USES, 'joints'),
                           # For now feet and joints are in same box
                           #(USES, 'feet'),
                           (CONSUMES, 'leg'),
                           (USES, 'screwdriver'),
                           (USES, 'screws'),
                           ]


class AssembleTopJoint(SupportedAction):

    hold = 'h'

    def __init__(self, leg):
        super(AssembleTopJoint, self).__init__('Assemble joint on ' + leg)
        self.conditions = [(USES, 'joints'),
                           (USES, 'screwdriver'),
                           (USES, 'screws'),
                           ]


class AssembleLegToTop(SupportedAction):

    hold = 'v'

    def __init__(self, leg, bring_top=False):
        super(AssembleLegToTop, self).__init__('Assemble {} to top'.format(leg))
        self.conditions = [(CONSUMES, 'top')] if bring_top else []
        self.conditions.extend([(USES, 'screwdriver'),
                                (USES, 'screws'),
                                ])


class _SupportivePOMDPState(object):
    """Enables conversion of states between integers and array representation
    """

    def __init__(self, n_htm_states, n_preferences, n_body_features, n_objects, s=0):
        self._shift_body = n_objects
        self._shift_pref = self._shift_body + n_body_features
        self._shift_htm = self._shift_pref + n_preferences
        self._final_htm = n_htm_states - 1
        self.s = s

    def __str__(self):
        return "<{: >{w}}: {} {} {} {}>".format(
            self.s,
            self.htm,
            "".join([str(self.has_preference(i))
                     for i in range(self._shift_htm - self._shift_pref)]),
            "".join([str(self.has_body_feature(i))
                     for i in range(self._shift_pref - self._shift_body)]),
            "".join([str(self.has_object(i))
                     for i in range(self._shift_body)]),
            w=len(str(self.n_states - 1)))

    @property
    def n_objects(self):
        return self._shift_body

    @property
    def n_preferences(self):
        return self._shift_htm - self._shift_pref

    @property
    def n_htm(self):
        return self._final_htm + 1

    @property
    def htm(self):
        return self.s >> self._shift_htm

    @htm.setter
    def htm(self, n):
        self.s += (n << self._shift_htm) - (self.htm << self._shift_htm)

    @property
    def n_states(self):
        return (self._final_htm + 1) * 2 ** (self._shift_htm)

    def is_final(self):
        return self.htm == self._final_htm

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

    def belief_quotient(self, array):
        """Assimilates all states that correspond to same HTM node."""
        n = 2 ** self._shift_htm
        assert(array.shape[0] == self.n_htm * n)
        return array.reshape((self.n_htm, n)).sum(axis=1)

    def belief_preferences(self, array):
        """Assimilates all states that correspond to same HTM node."""
        n = 2 ** self._shift_pref
        n_p = self.n_preferences
        assert(array.shape[0] == self.n_htm * (2 ** n_p) * n)
        new_shape = [self.n_htm] + [2 for _ in range(n_p)] + [n]
        pp = array.reshape(new_shape).sum(axis=-1).sum(axis=0)

        def sum_all_but(arr, axis):
            iii = [slice(None) for _ in arr.shape]
            iii[axis] = -1
            return arr[iii].sum()

        return [sum_all_but(pp, i) for i, _ in enumerate(pp.shape)]

    def random_object_changes(self, p):
        to_change = np.random.random((self.n_objects)) < p
        for i in to_change.nonzero()[0]:
            self._set_bit(i, 1 - self._get_bit(i))

    def random_preference_changes(self, p):
        to_change = np.random.random((self.n_preferences)) < p
        for i in to_change.nonzero()[0]:
            self.set_preference(i, 1 - self.has_preference(i))


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
    A_HOLD_H = 1
    A_HOLD_V = 2
    A_ASK = 3

    O_NONE = 0
    O_FAIL = 1
    O_NOT_FOUND = 2
    O_YES = 3
    O_NO = 4

    PREF_HOLD = 0

    p_fail = .01
    p_consume_all = .5
    p_changed_by_human = .05
    p_change_preference = .05

    r_subtask = 10.
    r_final = 100.
    r_preference = 10.

    cost_get = 15.
    cost_intrinsic = 1.
    cost_hold = 2.

    preferences = ['hold']
    p_preferences = [0.3]

    observations = ['none', 'fail', 'not-found', 'yes', 'no']
    n_observations = len(observations)

    def __init__(self, htm, discount=1.):
        self.discount = discount
        h2d = _HTMToDAG(htm)
        self.htm_nodes = h2d.nodes
        assert(len(self.htm_nodes) < 128)  # we use dtype=np.int8
        # set final state as successors of last actions in HTM
        self.htm_succs = [[self.htm_clean] if len(s) == 0 else s for s in h2d.succs]
        self.htm_init = h2d.init
        self._populate_conditions()
        self.n_states = self.n_htm_states * (
            2 ** (len(self.preferences) + 1 + len(self.objects)))
        self._init_object_actions_indices()
        self.n_actions = self._skip_to_a_obj + len(self.objects) + sum(self.clearable)

    def _int_to_state(self, s=0):
        return _SupportivePOMDPState(self.n_htm_states, len(self.preferences),
                                     1, len(self.objects), s=s)

    def get_object_id(self, object_name):
        # Note: this is not efficient
        if object_name not in self.objects:
            self.objects.append(object_name)
            self.clearable.append(False)
        return self.objects.index(object_name)

    def _populate_conditions(self):
        self.objects = []
        self.clearable = []
        self.htm_conditions = [[] for _ in self.htm_nodes]
        for n, conditions in zip(self.htm_nodes, self.htm_conditions):
            for c, o in n.action.conditions:
                i_o = self.get_object_id(o)
                if c == USES or c == CONSUMES_SOME:
                    self.clearable[i_o] = True
                conditions.append((c, i_o))

    @property
    def n_htm_states(self):
        """Number of nodes in the DAG plus 1 for the final state.
        """
        return len(self.htm_nodes) + 2

    @property
    def htm_clean(self):
        return len(self.htm_nodes)

    @property
    def htm_final(self):
        return len(self.htm_nodes) + 1

    def is_final(self, s):
        return self._int_to_state(s).is_final()

    @property
    def features(self):
        return (['HTM'] +
                ['{}-preference'.format(p) for p in self.preferences] +
                ['holding'] +
                self.objects)

    @property
    def states(self):
        return [str(self._int_to_state(i)) for i in range(self.n_states)]

    @property
    def htm_names(self):
        return [n.name for n in self.htm_nodes] + ['clean', 'final']

    @property
    def actions(self):
        return ['wait', 'hold H', 'hold V', 'ask hold'] + list(chain(*[
            ['bring ' + o] + (['clear ' + o] if c else [])
            for o, c in zip(self.objects, self.clearable)]))

    def belief_update(self, a, o, b):
        raise NotImplemented

    def _update_for_transition(self, _s, node):
        """Computes reward and modifies state to match transition from node
        to a random successor.
        """
        _s.htm = np.random.choice(self.htm_succs[node])
        return sum([self._update_for_condition(_s, c, o)
                    for c, o in self.htm_conditions[node]])

    # Action indices manipulation

    def _init_object_actions_indices(self):
        self._skip_to_a_obj = 4
        self._a_bring = [None for _ in self.objects]
        self._a_remove = [None for _ in self.objects]
        j = self._skip_to_a_obj
        for (o, c) in enumerate(self.clearable):
            self._a_bring[o] = j
            j += 1
            if c:
                self._a_remove[o] = j
                j += 1

    def _obj_from_action(self, a):
        if a in self._a_bring:
            return self._a_bring.index(a)
        elif a in self._a_remove:
            return self._a_remove.index(a)
        else:
            raise ValueError("Not an object action: %s" % a)

    def _is_bring(self, a):
        return a in self._a_bring

    def _bring(self, obj):
        return self._a_bring[obj]

    def _remove(self, obj):
        a = self._a_remove[obj]
        if a is None:
            raise ValueError('{} is not cleanable.'.format(obj))
        return a

    def _update_for_condition(self, _s, c, obj):
        """Computes reward and modifies state according to conditions.
        """
        r = 0. if _s.has_object(obj) else -self._cost_get(obj)
        # note: all conditions need object to be there
        if c == CONSUMES:
            _s.set_object(obj, 0)
        elif c == CONSUMES_SOME:
            _s.set_object(obj, 0 if np.random.random() < self.p_consume_all else 1)
        elif c == USES:
            _s.set_object(obj, 1)
        return r

    def _cost_get(self, o):
        return self.cost_get  # Cost for the human to get the object

    def sample_transition(self, a, s, random=True):
        _s = self._int_to_state(s)
        if random:
            # random transitions
            _s.random_object_changes(self.p_changed_by_human)
            _s.random_preference_changes(self.p_change_preference)
        _new_s = self._int_to_state(_s.to_int())
        if a in (self.A_WAIT, self.A_HOLD_H, self.A_HOLD_V):
            r = 0 if a == self.A_WAIT else -self.cost_hold
            if _s.is_final():  # Final state
                obs = self.O_NONE
            elif _s.htm == self.htm_clean:
                obs = self.O_NONE
                for o, _ in enumerate(self.objects):
                    r -= self._cost_get(o) * _s.has_object(o)
                _new_s.htm = self.htm_final
                r += self.r_final
            else:
                obs = self.O_NONE
                needs_hold = self.htm_nodes[_s.htm].action.hold
                if _s.has_preference(self.PREF_HOLD) and (
                        (needs_hold == 'h' and a == self.A_HOLD_H) or
                        (needs_hold == 'v' and a == self.A_HOLD_V)):
                    r += self.r_preference
                elif (a in (self.A_HOLD_H, self.A_HOLD_V) and (
                        (not random) or np.random.random() < .95)):
                    # Undesired hold most likely gets an error
                    obs = self.O_FAIL
                r += self._update_for_transition(_new_s, _s.htm)
                r += self.r_subtask
        elif a == self.A_ASK:
            r = -self.cost_intrinsic
            if _s.has_preference(self.PREF_HOLD):
                if (not random) or np.random.random() < 0.9:
                    obs = self.O_YES
                else:
                    obs = self.O_NONE
            else:
                if (not random) or np.random.random() < 0.95:
                    obs = self.O_NO
                else:
                    obs = self.O_NONE
        else:
            obj = self._obj_from_action(a)
            is_bring = int(self._is_bring(a))
            if _s.has_object(obj) == is_bring:
                # Bring object already there or remove object that's not there
                obs = self.O_NOT_FOUND
            elif random and np.random.random() < self.p_fail:
                obs = self.O_FAIL
            else:
                _new_s.set_object(obj, is_bring)
                obs = self.O_NONE
            # TODO: add random transitions on other features
            r = -self.cost_intrinsic  # Intrinsic action cost
        return _new_s.to_int(), obs, r

    def sample_start(self):
        htm_id = np.random.choice(self.htm_init)
        _s = self._int_to_state()
        _s.htm = htm_id
        for i, p in enumerate(self.p_preferences):
            _s.set_preference(i, 1 if np.random.random() < p else 0)
        # random transitions
        _s.random_object_changes(self.p_changed_by_human)
        return _s.to_int()

    @property
    def start(self):
        raise NotImplementedError


class NHTMHorizon(Horizon):

    def __init__(self, model, n):
        self.model = model
        self.n = n

    def is_reached(self):
        return self.n <= 0

    def decrement(self, a, s, new_s, o):
        _new_s = self.model._int_to_state(new_s)
        if self.model._int_to_state(s).htm != _new_s.htm:
            self.n -= 1
        if _new_s.is_final():
            self.n = 0

    def copy(self):
        return NHTMHorizon(self.model, self.n)

    @classmethod
    def generator(cls, model, n=3):
        return cls._Generator(cls, model, n)
