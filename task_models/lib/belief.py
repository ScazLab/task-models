import numpy as np

from .utils import assert_normal


class BaseBelief(object):

    def sample(self):
        raise NotImplemented

    def successor(self, model, a, o):
        raise NotImplemented

    def to_list(self):
        return self.array.tolist()


class ArrayBelief(BaseBelief):

    def __init__(self, probabilities):
        self.array = np.asarray(probabilities)
        assert_normal(self.array, name='probabilities')

    def __hash__(self):
        return hash(self.array.tostring())

    def __eq__(self, other):
        return (isinstance(other, ArrayBelief) and
                (self.array == other.array).all())

    def sample(self):
        return np.random.choice(self.array.shape[0], p=self.array)

    def successor(self, model, a, o):
        return ArrayBelief(model.belief_update(a, o, self.array))


class MaxSamplesReached(RuntimeError):

    default_msg = "Impossible to sample enough particles."

    def __init__(self, a, o, b, msg=None):
        self.belief = b
        self.action = a
        self.observation = o
        self.msg = self.default_msg if msg is None else msg

    def __str__(self):
        return "{} (transition with action {} and observation {})".format(
            self.msg, self.action, self.observation)


class _SuccessorSampler:

    def __init__(self, model, belief, a, o, max_samples=1000):
        self.n_sampled = 0
        self.model = model
        self.belief = belief
        self.a = a
        self.o = o
        self.max_samples = max_samples

    def _sample(self):
        self.n_sampled += 1
        if self.n_sampled > self.max_samples:
            raise MaxSamplesReached(self.a, self.o, self.belief)
        return self.belief.sample()

    def __call__(self):
        o = None
        while o != self.o:
            s, o, _ = self.model.sample_transition(self.a, self._sample())
        return s


class ParticleBelief(BaseBelief):

    def __init__(self, sampler, n_states, n_particles=100):
        self.n_states = n_states
        self.n_particles = n_particles
        self.part_states = []
        self._populate(sampler)

    def _populate(self, sampler):
        try:
            while len(self.part_states) < self.n_particles:
                self.part_states.append(sampler())
        except MaxSamplesReached as e:
            if len(self.part_states) > 0:
                # duplicate found particles
                n_part = len(self.part_states)

                def self_sampler():
                    return self.sample(_max_index=n_part)

                self._populate(self_sampler)
            else:
                e.msg = "Impossible to sample any particle."
                raise e

    def sample(self, _max_index=None):
        if _max_index is None:
            _max_index = self.n_particles
        return self.part_states[np.random.choice(_max_index)]

    def successor(self, model, a, o):
        sampler = _SuccessorSampler(model, self, a, o,
                                    max_samples=100 * self.n_particles)
        return ParticleBelief(sampler, self.n_states, self.n_particles)

    @property
    def array(self):
        a = np.zeros((self.n_states))
        for i in self.part_states:
            a[i] += 1
        return a / (a.sum() or 1.)


def _format_p(x):
    s = "{:0.1f}".format(x)
    return "1." if s == "1.0" else s[1:]


def format_belief_array(b):  # b is an array
    return " ".join([_format_p(p) for p in b])
