import numpy as np

from htm.lib.pomdp import POMDP
from htm.task import (AbstractAction, SequentialCombination,
                      AlternativeCombination, LeafCombination)


def human_proba_from_time(t, t_human_action_avg):
    return np.exp(-t / t_human_action_avg)


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


class HTM2POMDP:

    observations = ['none', 'yes', 'no']

    def __init__(self, t_wait, t_com):
        self.t_wait = t_wait
        self.t_com = t_com

    def _name_radix(self, action):
        return action.name.lower().replace(' ', '-')

    def _leaf_to_pomdp(self, leaf):
        t_hum, t_rob, t_err = leaf.action.durations
        radix = self._name_radix(leaf.action)
        # states: before (H: human intend act), before (R: robot, human do not
        # intend to act), after
        states = [n + radix for n in ["before-H-", "before-R-", "afer-"]]
        # actions: wait (shared), physical, communicate
        actions = ["wait"] + [n + radix for n in ["act-", "com-"]]
        init = [0, 1]  # initial states
        end = 2  # ending state (unique, will be removed)
        start = np.array([.5, .5]) if leaf.action.h_proba is None \
            else np.array(leaf.action.human_probability)
        p_wait = human_proba_from_time(self.t_wait, t_hum)
        p_phys = human_proba_from_time(t_err, t_hum)
        p_comm = human_proba_from_time(self.t_com, t_hum)
        # Note: if error the model assumes that human waits that robot is done
        # recovering or communicating before moving to next action.
        T = np.zeros((3, 3, 3))
        T[:, 0, :] = [[p_wait, 0., 1-p_wait],
                      [p_phys, 0., 1-p_phys],
                      [p_comm, 0., 1-p_comm]]
        T[:, 1, :] = [[0.,     1.,       0.],
                      [0.,     0.,       1.],
                      [0.,     1.,       0.]]
        T[:, 2, :] = [0.,     0.,       1.]
        O = np.zeros((3, 3, 3))
        O[0, :, :] = [1., 0., 0.]  # Always observe nothing on wait
        O[1, :, :] = [[0., 0., 1.],  # error observed
                      [1., 0., 0.],
                      [0., 0., 1.]]  # error observed
        O[2, :, :] = [[0., 0., 1.],  # Human always answer, even
                      [0., 0., 1.],  # when it's the robot who
                      [0., 1., 0.]]  # completed the action
        R = np.zeros((3, 3, 3, 3))
        # Convention: no cost in final state
        R[0, :-1, :, :] = self.t_wait
        R[1, 0, :, :] = t_err
        R[1, 1, :, :] = t_rob
        R[2, :-1, :, :] = self.t_com
        # TODO: also return
        # - how to compute human transitions on other actions
        # - default cost on other states for actions
        # - observations on other actions
        # - observations on other states (before and after)
        return (T, O, R, start, states, actions, init, end)

    def _sequence_to_pomdp(self, seq):
        raise NotImplementedError

    def _alternative_to_pomdp(self, alt):
        raise NotImplementedError

    def _node_to_pomdp(self, node):
        if isinstance(node, LeafCombination):
            return self._leaf_to_pomdp(node)
        elif isinstance(node, SequentialCombination):
            return self._sequence_to_pomdp(node)
        elif isinstance(node, AlternativeCombination):
            return self._alternatives_to_pomdp(node)
        else:
            raise ValueError('Unkown combination: ' + type(node))

    def task_to_pomdp(self, task):
        T, O, R, pstart, states, actions, init, end = self._node_to_pomdp(
            task.root)
        start = np.zeros(len(states))
        start[init] = pstart
        return POMDP(T, O, R, start, discount=1., states=states,
                     actions=actions, observations=self.observations,
                     values='cost')
