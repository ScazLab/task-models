from unittest import TestCase

import numpy as np

from task_models.lib.multiprocess import repeat


class TestRepeat(TestCase):

    def target(self):
        return 42

    def test_repeat_once(self):
        self.assertEqual(repeat(self.target, 1), [42])

    def test_repeat_many(self):
        repeated = repeat(self.target, 5)
        self.assertEqual(len(repeated), 5)
        self.assertTrue(all([x == 42 for x in repeated]))


class TestRandomness(TestCase):

    def get_random(self):
        return np.random.randint(10)

    def test_subprocess_get_different_randoms(self):
        randoms = repeat(self.get_random, 10)
        self.assertTrue(len(set(randoms)) > 1)
