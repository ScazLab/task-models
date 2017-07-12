import signal
from unittest import TestCase

import numpy as np

from multiprocess import repeat


class _TimeoutContext:

    def __init__(self, timeout):
        self.timeout = timeout

    def timeoutFail(self, signum, frame):
        raise AssertionError('Timeout in assertion')

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeoutFail)
        signal.alarm(self.timeout)

    def __exit__(self, *args):
        signal.alarm(0)


class TestRepeat(TestCase):

    def target(self):
        return 42

    def assertFinishesInTime(self, timeout=1):
        return _TimeoutContext(timeout)

    def test_repeat_once(self):
        self.assertEqual(repeat(self.target, 1), [42])

    def test_repeat_many(self):
        repeated = repeat(self.target, 5)
        self.assertEqual(len(repeated), 5)
        self.assertTrue(all([x == 42 for x in repeated]))

    def test_repeat_large_data(self):
        """Non regression test: deadlock on full pipe if queue not emptied."""

        def more_results():
            return ["abcd" * 1000] * 1000

        with self.assertFinishesInTime():
            repeat(more_results, 100)


class TestRandomness(TestCase):

    def get_random(self):
        return np.random.randint(10)

    def test_subprocess_get_different_randoms(self):
        randoms = repeat(self.get_random, 10)
        self.assertTrue(len(set(randoms)) > 1)
