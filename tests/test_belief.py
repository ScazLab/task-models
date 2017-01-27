from numbers import Integral
from unittest import TestCase

import numpy as np

from htm.lib.belief import ArrayBelief, ParticleBelief


class BeliefBaseTest(object):

    class FakeModel:

        def __init__(self, successor_0, successor_1):
            self.sampled_on = []
            self.successors = [successor_0, successor_1]

        def belief_update(self, a, o, array):
            return self.successors[o]  # only accepts o = (0|1)

        def sample_transition(self, a, s):
            self.sampled_on.append((a, s))
            if np.random.random() < .5:
                o = 0
            else:
                o = 1
            return np.random.choice(3, p=self.successors[o]), o, -1.5

    def setUp(self):
        self.p = np.array([.7, 0., .3])

    def test_sample_is_int(self):
        self.assertIsInstance(self.belief.sample(), Integral)

    def test_sample_only_from_non_zeros(self):
        for i in range(10):
            self.assertNotEqual(self.p[self.belief.sample()], 0.)


class TestArrayBelief(BeliefBaseTest, TestCase):

    def setUp(self):
        super(TestArrayBelief, self).setUp()
        self.belief = ArrayBelief(self.p)

    def test_successor(self):
        p_succ = np.array([.6, .4, 0.])
        model = self.FakeModel(np.array([0., 0.1, 0.9]), p_succ)
        succ = self.belief.successor(model, 2, 1)
        self.assertIsInstance(succ, ArrayBelief)
        np.testing.assert_array_equal(succ.array, p_succ)


class TestParticleBelief(BeliefBaseTest, TestCase):

    def setUp(self):
        super(TestParticleBelief, self).setUp()

        def sampler():
            return int(np.random.choice(3, p=self.p))

        self.belief = ParticleBelief(sampler, 3, 10)

    def test_successor(self):
        p_succ = np.array([.6, .4, 0.])
        model = self.FakeModel(np.array([0., 0.1, 0.9]), p_succ)
        succ = self.belief.successor(model, 2, 1)
        self.assertIsInstance(succ, ParticleBelief)
        self.assertNotIn(2, succ.part_states)
        self.assertTrue(all([a == 2 for (a, _) in model.sampled_on]))
        self.assertTrue(all([s in self.belief.part_states
                             for (_, s) in model.sampled_on]))
