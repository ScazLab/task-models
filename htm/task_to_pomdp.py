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

    observations = ['none', 'yes', 'no']

    o_none = [1., 0., 0.]
    o_yes = [0., 1., 0.]
    o_no = [0., 0., 1.]

    init = None     # index of init states
    start = None    # start probabilities
    states = None   # state names
    actions = None  # action names
    a_act = None    # list of physical actions
    a_com = None    # list of communication actions

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

    def update_O(self, O, a_wait, a_start, s_start, a_comm_before,
                 a_comm_after, a_act):
        """Fills relevant parts of O.
        Every node is responsible for filling O[:, [one own's states], :].
        O is assumed to be initialized with zeros.
        :param a_comm_before: list of indices
            indices of communication act about actions that are a prerequisite
            of current node (does not include current nodes actions).
            i.e. questions to which answer 'yes'
        :param a_comm_after: list of indices
            indices other all other actions except those of current node
        :param a_act: list of indices
            indices of physical actions, includes current node's actions
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

    init = [0, 1]  # initial states
    end = 2  # ending state (unique, will be removed)
    _act = 0
    a_act = [_act]
    _com = 1
    a_com = [_com]

    act = 'phy'
    com = 'com'

    def __init__(self, leaf, t_com):
        self.t_com = t_com
        self.leaf = leaf
        # states: before (H: human intend act), before (R: robot, human do not
        # intend to act), after
        radix = _name_radix(leaf.action)
        self.states = [n + radix for n in ["before-H-", "before-R-"]]
        # actions: wait (shared), physical, communicate
        self.actions = [n + '-' + radix for n in [self.act, self.com]]

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
        return [self.t_err, self.t_com]  # does not include wait

    @property
    def p_h(self):
        return .5 if self.leaf.action.h_proba is None\
            else self.leaf.action.h_proba

    @property
    def start(self):
        return [self.p_h, 1-self.p_h]

    def _h_probas_not_finished_from_d(self, durations):
        return np.exp(-durations / self.t_hum)

    def update_T(self, T, a_wait, a_start, s_start, s_next, s_next_probas,
                 durations):
        assert(T.shape[0] == len(durations))
        p_h_not_finish = self._h_probas_not_finished_from_d(
                np.asarray(durations))
        # Note: if error the model assumes that human waits that robot is done
        # recovering or communicating before moving to next action.
        T[:, s_start, s_start] = p_h_not_finish
        T[:, s_start, :][:, s_next] = \
            np.outer(1 - p_h_not_finish, s_next_probas)
        T[:, s_start + 1, s_start + 1] = 1.
        T[a_start + self._act, s_start + 1, s_start + 1] = 0
        T[a_start + self._act, s_start + 1, :][s_next] = s_next_probas

    def update_O(self, O, a_wait, a_start, s_start, a_comm_before,
                 a_comm_after, a_act):
        O[a_wait, s_start:s_start+2, :] = self.o_none
        # (always observe nothing on wait)
        for a in a_act:
            O[a, s_start:s_start+2, :] = self.o_no  # wrong actions
        O[a_start + self._act, s_start:s_start+2, :] = [
            self.o_no,    # good action but human is acting
            self.o_none]  # good action
        for a in a_comm_before:
            O[a, s_start:s_start+2, :] = self.o_yes
        # (human always answer, even when it's the robot who completed
        #  the action)
        for a in a_comm_after + [a_start + self._com]:
            O[a, s_start:s_start+2, :] = self.o_no

    def update_R(self, R, a_wait, a_start, s_start, durations):
        # Note: every node is responsible for filling
        # R[:, [one own's states], :, :]
        # R is initialized with zeros.
        R[:, s_start:s_start+2, :, :] = np.asarray(durations)[
            :, np.newaxis, np.newaxis, np.newaxis]
        R[a_start + self._act, s_start+1, :, :] = self.t_rob


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

    def _shift_children_actions(self, action_lists, node_a_start=0):
        return [[node_a_start + a_start + a for a in l]
                for a_start, l in zip(self.a_indices, action_lists)]

    @property
    def a_act(self):
        return sum(self._shift_children_actions(
            [c.a_act for c in self.children]),
            [])

    @property
    def a_com(self):
        return sum(self._shift_children_actions(
            [c.a_com for c in self.children]),
            [])

    def update_T(self, T, a_wait, a_start, s_start, s_next, s_next_probas,
                 durations):
        next_inits = [
            [s_start + c_s_start + s for s in c.init]
            for c, c_s_start in zip(self.children[1:], self.s_indices[1:])]
        next_inits += [s_next]
        next_probas = [c.start for c in self.children[1:]] + [s_next_probas]
        for i, c in enumerate(self.children):
            c.update_T(T, a_wait, a_start + self.a_indices[i],
                       s_start + self.s_indices[i], next_inits[i],
                       next_probas[i], durations)

    def update_O(self, O, a_wait, a_start, s_start, a_comm_before,
                 a_comm_after, a_act):
        a_coms = self._shift_children_actions([c.a_com for c in self.children],
                                              node_a_start=a_start)
        for i, c in enumerate(self.children):
            ac_before = sum(a_coms[:i], [])
            ac_after = sum(a_coms[i+1:], [])
            c.update_O(O, a_wait, a_start + self.a_indices[i],
                       s_start + self.s_indices[i],
                       a_comm_before + ac_before, a_comm_after + ac_after,
                       a_act)

    def update_R(self, R, a_wait, a_start, s_start, durations):
        for i, c in enumerate(self.children):
            c.update_R(R, a_wait, a_start + self.a_indices[i],
                       s_start + self.s_indices[i], durations)


class _AlternativesToPOMDP(_NodeToPOMDP):

    init = None     # index of init states
    start = None    # start probabilities
    states = None   # state names
    actions = None  # action names
    a_act = None    # list of physical actions
    a_com = None    # list of communication actions

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

    def update_O(self, O, a_wait, a_start, s_start, a_comm_before,
                 a_comm_after, a_act):
        """Fills relevant parts of O.
        Every node is responsible for filling O[:, [one own's states], :].
        O is assumed to be initialized with zeros.
        :param a_comm_before: list of indices
            indices of communication act about actions that are a prerequisite
            of current node (does not include current nodes actions).
            i.e. questions to which answer 'yes'
        :param a_comm_after: list of indices
            indices other all other actions except those of current node
        :param a_act: list of indices
            indices of physical actions, includes current node's actions
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

    def update_O_end(self, O):
        O[:, self.end, :] = _NodeToPOMDP.o_none
        # nothing is observed in end state

    def update_R_end(self, R):
        R[:, -1, ...] = 0.  # end state has no cost

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
        n2p.update_O(O, self.wait, 1, 0, [], [], [a + 1 for a in n2p.a_act])
        self.update_O_end(O)
        R = np.zeros((n_a, n_s, n_s, n_o))
        n2p.update_R(R, self.wait, 1, 0, durations)
        self.update_R_end(R)
        return POMDP(T, O, R, start, discount=1., states=states,
                     actions=actions, observations=n2p.observations,
                     values='cost')
