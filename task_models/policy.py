import numpy as np


class PolicyRunner(object):

    class UnexpectedObservation(RuntimeError):
        pass

    def __init__(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, o):
        raise Exception(o)

    @property
    def actions(self):
        return self.model.actions

    @property
    def observations(self):
        return self.model.observations

    def reset(self):
        self.history = []

    def get_action(self):
        """Needs to store action index as self._last_action."""
        raise NotImplementedError

    def step(self, observation):
        if self._last_action is None:
            raise ValueError('Unknown last action')
        o = self.observations.index(observation)
        self.history.extend([self._last_action, o])


class PolicyLongSupportiveSequence(PolicyRunner):

    def __init__(self, model):
        super(PolicyLongSupportiveSequence, self).__init__(model)
        self._a_wait = self.actions.index('wait')
        self._o_none = self.observations.index('none')
        self.n_tasks = self.model.n_htm_states - 2
        self.tools = ['screws', 'screwdriver', 'joints']
        self._a_next_list = [self.actions.index('hold H')]
        self.reset()

    @property
    def _a_bring_leg(self):
        return self._a_bring('leg')

    @property
    def _a_next(self):
        return np.random.choice(self._a_next_list)

    def _is_a_next(self, a):
        return a in self._a_next_list

    def _a_bring(self, obj):
        return self.actions.index('bring {}'.format(obj))

    def _a_clean(self, obj):
        return self.actions.index('clear {}'.format(obj))

    def reset(self):
        super(PolicyLongSupportiveSequence, self).reset()
        self._to_bring = [self._a_bring(o) for o in self.tools]
        self._to_clean = [self._a_clean(o) for o in self.tools]
        self._needs_leg = True
        self._last_action = None
        self._n_done = 0

    def get_action(self, iterations=None):
        if len(self._to_bring) > 0:
            a = self._to_bring[0]
        elif self._n_done < self.n_tasks:
            if self._needs_leg:
                a = self._a_bring_leg
            else:
                a = self._a_next
        elif len(self._to_clean) > 0:
            a = self._to_clean[0]
        else:  # Final wait
            a = self._a_wait
        self._last_action = a
        return self.actions[a]

    def step(self, observation):
        super(PolicyLongSupportiveSequence, self).step(observation)
        unexpected = self.UnexpectedObservation('{} for action {}'.format(
            observation, self.actions[self._last_action]))
        if observation == 'fail':
            pass
        elif observation in ('none', 'not-found'):
            if self._last_action in [self._a_bring(o) for o in self.tools]:
                self._to_bring.remove(self._last_action)
            elif self._last_action in [self._a_clean(o) for o in self.tools]:
                self._to_clean.remove(self._last_action)
            elif self._last_action == self._a_bring_leg:
                self._needs_leg = False
            elif self._is_a_next(self._last_action) and observation == 'none':
                # Next subtask
                self._n_done += 1
                self._needs_leg = True
            elif self._last_action == self._a_wait:
                pass
            else:
                raise unexpected
        else:
            raise unexpected
