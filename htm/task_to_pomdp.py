import numpy as np

from htm.lib.pomdp import POMDP
from htm.task import (AbstractAction, SequentialCombination,
                      AlternativeCombination, LeafCombination)


class CollaborativeAction(AbstractAction):
    """Collaborative action that can be achieved either by robot or human.

    :param name: str (must be unique)
    :param durations: (float, float, float)
        Duration for robot and human to perform the action as well as error
        time when robot starts action at wrong moment (the error time includes
        the time of acting before interruption as well as the recovery time).
    :param human_probability: float
        Probability that the human would take care of this action. If not
        defined, will have to be estimated.
    """

    def __init__(self, name, durations, human_probability=None):
        super(CollaborativeAction, self).__init__(name=name)
        self.durations = durations
        self.h_proba = human_probability


def _name_radix(action):
    return action.name.lower().replace(' ', '-')


class _NodeToPOMDP(object):

    observations = ['none', 'yes', 'no', 'error']

    o_none = [1., 0., 0., 0.]
    o_yes = [0., 1., 0., 0.]
    o_no = [0., 0., 1., 0.]
    o_err = [0., 0., 0., 1.]

    init = None     # index of init states
    start = None    # start probabilities
    states = None   # state names
    actions = None  # action names

    @property
    def durations(self):
        """List of durations for node actions.

        Must include recovery time for erroneous physical actions.
        """
        raise NotImplementedError

    def update_T(self, T, a_wait, a_start, s_start, s_next, s_next_probas,
                 durations):
        """Fills relevant parts of T.
        Every node is responsible for filling T[:, [one own's states], :].
        T is assumed to be initialized with zeros.
        """
        raise NotImplementedError

    def update_O(self, O, a_start, s_start, s_next, s_before, s_after):
        """Fills relevant parts of O.
        Every node is responsible for filling
            O[[own's actions], :, :]
            Initialized with deterministic NONE.
        :param s_before: list of indices
            indices of states that are a prerequisite of current node (does
            not include current nodes actions).
        :param s_after: list of indices
            indices of states that are guaranteed to occur after current's
        """
        raise NotImplementedError

    def update_R(self, R, a_wait, a_start, s_start, durations):
        """Fills relevant parts of R.
        Every node is responsible for filling R[:, [one own's states], :, :].
        R is assumed to be initialized with zeros.
        """
        raise NotImplementedError

    @staticmethod
    def from_node(node, t_com):
        if isinstance(node, LeafCombination):
            return _LeafToPOMDP(node, t_com)
        elif isinstance(node, SequentialCombination):
            return _SequenceToPOMDP(node, t_com)
        elif isinstance(node, AlternativeCombination):
            return _AlternativesToPOMDP(node, t_com)
        else:
            raise ValueError('Unkown combination: ' + type(node))


class _LeafToPOMDP(_NodeToPOMDP):

    _init = 0  # Initial state
    _h = 1     # "Human has declared intent to act" state
    _r = 2     # "Robot has declared intent to act" state
    init = [_init]  # initial states
    start = [1.]
    _phy = 0
    _ask_intention = 1
    _tell_intention = 2
    _ask_finished = 3

    act = 'phy'
    com = 'com'

    def __init__(self, leaf, t_com):
        self.t_com = t_com
        self.leaf = leaf
        # states: before (H: human intend act), before (R: robot, human do not
        # intend to act), after
        radix = _name_radix(leaf.action)
        self.states = [n + radix for n in ["init-", "H-", "R-"]]
        # actions: wait (shared), physical, communicate
        self.actions = [n + '-' + radix
                        for n in [self.act, self.com + '-ask-intention',
                                  self.com + '-tell-intention',
                                  self.com + '-ask-finished']]

    @property
    def t_hum(self):
        return self.leaf.action.durations[0]

    @property
    def t_rob(self):
        return self.leaf.action.durations[1]

    @property
    def t_err(self):
        return self.leaf.action.durations[2]

    @property
    def durations(self):
        return [self.t_err] + [self.t_com] * 3  # does not include wait

    def _h_probas_not_finished_from_d(self, durations):
        return np.exp(-durations / self.t_hum)

    def update_T(self, T, a_wait, a_start, s_start, s_next, s_next_probas,
                 durations):
        assert(T.shape[0] == len(durations))
        a_phy = a_start + self._phy
        a_ai = a_start + self._ask_intention
        a_ti = a_start + self._tell_intention
        s_i = s_start + self._init
        s_h = s_start + self._h
        s_r = s_start + self._r
        # Note: if error the model assumes that human waits that robot is done
        # recovering or communicating before moving to next action.
        T[:, s_i, s_i] = 1.
        T[a_ai, s_i, s_i] = 0.
        T[a_ai, s_i, s_h] = 1.
        T[a_ti, s_i, s_i] = 0.
        T[a_ti, s_i, s_r] = 1.
        p_h_not_finish = self._h_probas_not_finished_from_d(
                np.asarray(durations))
        T[:, s_h, s_h] = p_h_not_finish
        T[:, s_h, :][:, s_next] = \
            np.outer(1 - p_h_not_finish, s_next_probas)
        T[:, s_r, s_r] = 1.
        T[a_phy, s_r, s_r] = 0.
        T[a_phy, s_r, s_next] = s_next_probas

    def update_O(self, O, a_start, s_start, s_next, s_before, s_after):
        a_phy = a_start + self._phy
        a_ai = a_start + self._ask_intention
        a_ti = a_start + self._tell_intention
        a_af = a_start + self._ask_finished
        s_h = s_start + self._h
        s_r = s_start + self._r
        O[a_phy, :, :] = self.o_err
        O[a_phy, s_next, :] = self.o_none
        O[a_ai, :, :] = self.o_none
        O[a_ai, s_h, :] = self.o_yes
        O[a_ai, s_r, :] = self.o_no
        O[a_ti, :, :] = self.o_none
        O[a_af, :, :] = self.o_no
        O[a_af, s_after, :] = self.o_yes

    def update_R(self, R, a_wait, a_start, s_start, durations):
        # Note: every node is responsible for filling
        # R[:, [one own's states], :, :]
        # R is initialized with zeros.
        R[:, s_start:s_start+3, :, :] = np.asarray(durations)[
            :, np.newaxis, np.newaxis, np.newaxis]
        R[a_start + self._phy, s_start + self._r, :, :] = self.t_rob


def _start_indices_from(l):
    """:param l: list of lists
          list of each child's states or action
    """
    return np.cumsum([0] + [len(x) for x in l[:-1]])


class _SequenceToPOMDP(_NodeToPOMDP):

    def __init__(self, node, t_com):
        self.children = [self.from_node(n, t_com) for n in node.children]
        child_states = [c.states for c in self.children]
        self.s_indices = _start_indices_from(child_states)
        self.states = sum(child_states, [])
        child_actions = [c.actions for c in self.children]
        self.a_indices = _start_indices_from(child_actions)
        self.actions = sum(child_actions, [])

    @property
    def init(self):
        return self.children[0].init

    @property
    def durations(self):
        return sum([c.durations for c in self.children], [])

    @property
    def start(self):
        return self.children[0].start

    def _next_init_children(self, s_start, s_next):
        next_inits = [
            [s_start + c_s_start + s for s in c.init]
            for c, c_s_start in zip(self.children[1:], self.s_indices[1:])]
        next_inits += [s_next]
        return next_inits

    def update_T(self, T, a_wait, a_start, s_start, s_next, s_next_probas,
                 durations):
        next_inits = self._next_init_children(s_start, s_next)
        next_probas = [c.start for c in self.children[1:]] + [s_next_probas]
        for i, c in enumerate(self.children):
            c.update_T(T, a_wait, a_start + self.a_indices[i],
                       s_start + self.s_indices[i], next_inits[i],
                       next_probas[i], durations)

    def update_O(self, O, a_start, s_start, s_next, s_before, s_after):
        next_inits = self._next_init_children(s_start, s_next)
        states = [[self.s_indices[i] + j for j in range(len(c.states))]
                  for i, c in enumerate(self.children)]
        for i, c in enumerate(self.children):
            cs_before = sum(states[:i], [])
            cs_after = sum(states[i+1:], [])
            c.update_O(O, a_start + self.a_indices[i],
                       s_start + self.s_indices[i], next_inits[i],
                       s_before + cs_before, s_after + cs_after)

    def update_R(self, R, a_wait, a_start, s_start, durations):
        for i, c in enumerate(self.children):
            c.update_R(R, a_wait, a_start + self.a_indices[i],
                       s_start + self.s_indices[i], durations)


class _AlternativesToPOMDP(_NodeToPOMDP):

    init = None     # index of init states
    start = None    # start probabilities
    states = None   # state names
    actions = None  # action names

    @property
    def durations(self):
        """List of durations for node actions.

        Must include recovery time for erroneous physical actions.
        """
        raise NotImplementedError

    def update_T(self, T, a_wait, a_start, s_start, s_next, s_next_probas,
                 durations):
        """Fills relevant parts of T.
        Every node is responsible for filling T[:, [one own's states], :].
        T is assumed to be initialized with zeros.
        """
        raise NotImplementedError

    def update_O(self, O, a_start, s_start, s_next, s_before, s_after):
        """Fills relevant parts of O.
        Every node is responsible for filling O[:, [one own's states], :].
        O is assumed to be initialized with zeros.
        :param a_comm_before: list of indices
            indices of communication act about actions that are a prerequisite
            of current node (does not include current nodes actions).
            i.e. questions to which answer 'yes'
        :param a_comm_after: list of indices
            indices other all other actions except those of current node
        """
        raise NotImplementedError

    def update_R(self, R, a_wait, a_start, s_start, durations):
        """Fills relevant parts of R.
        Every node is responsible for filling R[:, [one own's states], :, :].
        R is assumed to be initialized with zeros.
        """
        raise NotImplementedError


class HTMToPOMDP:

    wait = 0
    end = -1

    def __init__(self, t_wait, t_com):
        self.t_wait = t_wait
        self.t_com = t_com

    def update_T_end(self, T):
        T[:, self.end, self.end] = 1.  # end stats is stable

    def update_O_wait(self, O):
        O[self.wait, :, :] = _NodeToPOMDP.o_none

    def update_R_end(self, R):
        R[:, -1, ...] = 1.          # end state has cost 1
        R[self.wait, -1, ...] = 0.  # except on wait

    def task_to_pomdp(self, task):
        n2p = _NodeToPOMDP.from_node(task.root, self.t_com)
        states = n2p.states + ['end']
        actions = ['wait'] + n2p.actions
        start = np.zeros(len(states))
        start[n2p.init] = n2p.start
        n_s = len(states)
        n_a = len(actions)
        n_o = len(n2p.observations)
        end = n_s - 1
        durations = [self.t_wait] + n2p.durations
        T = np.zeros((n_a, n_s, n_s))
        n2p.update_T(T, self.wait, 1, 0, [end], [1.], durations)
        self.update_T_end(T)
        O = np.zeros((n_a, n_s, n_o))
        n2p.update_O(O, 1, 0, [end], [], [end])
        self.update_O_wait(O)
        R = np.zeros((n_a, n_s, n_s, n_o))
        n2p.update_R(R, self.wait, 1, 0, durations)
        self.update_R_end(R)
        return POMDP(T, O, R, start, discount=1., states=states,
                     actions=actions, observations=n2p.observations,
                     values='cost')
