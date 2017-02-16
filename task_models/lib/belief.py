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
        return (isinstance(other, ArrayBelief)
                and (self.array == other.array).all())

    def sample(self):
        return np.random.choice(self.array.shape[0], p=self.array)

    def successor(self, model, a, o):
        return ArrayBelief(model.belief_update(a, o, self.array))


class MaxSamplesReached(RuntimeError):
    pass


class _SuccessorSampler:

    def __init__(self, model, belief, a, o, max_samples=100000):
        self.n_sampled = 0
        self.model = model
        self.belief = belief
        self.a = a
        self.o = o
        self.max_samples = max_samples

    def _sample(self):
        self.n_sampled += 1
        if self.n_sampled > self.max_samples:
            raise MaxSamplesReached(
                'Impossible to sample enough particles for transition to '
                + str((self.a, self.o)))
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
        self.part_states = [sampler() for _ in range(self.n_particles)]

    def sample(self):
        return self.part_states[np.random.choice(self.n_particles)]

    def successor(self, model, a, o):
        sampler = _SuccessorSampler(model, self, a, o,
                                    max_samples=1000 * self.n_particles)
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
