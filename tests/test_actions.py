from unittest import TestCase
import numpy as np

from task_models.state import NDimensionalState
from task_models.action import Condition, PrePostConditionAction


class TestCondition(TestCase):

    n_dim = 5

    def test_raises_value_error_on_masked_nonzeros(self):
        mask = np.array([0, 1, 0, 0, 1])
        value = np.array([1, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            Condition(mask, value)

    def test_check_true(self):
        mask = np.array([0, 1, 0, 0, 1])
        value = np.array([0, 0, 0, 0, -1])
        state_feat = np.random.random((5,))
        state_feat[4] = -1
        state_feat[1] = 0
        state = NDimensionalState(state_feat)
        self.assertTrue(Condition(mask, value).check(state))

    def test_check_false(self):
        mask = np.array([0, 1, 0, 0, 1])
        value = np.array([0, 0, 0, 0, -1])
        state_feat = np.random.random((5,))
        state_feat[4] = -1
        state_feat[1] = 1
        state = NDimensionalState(state_feat)
        self.assertFalse(Condition(mask, value).check(state))


class TestPrePostConditionAction(TestCase):

    def setUp(self):
        mask = np.array([0, 1, 0, 0, 1])
        value = np.array([0, 0, 0, 0, -1])
        self.pre = Condition(mask, value)
        mask = np.array([1, 0, 0, 0, 0])
        value = np.array([1, 0, 0, 0, 0])
        self.post = Condition(mask, value)
        state_feat = np.random.random((5,))
        state_feat[4] = -1
        state_feat[1] = 0
        self.before = NDimensionalState(state_feat.copy())
        state_feat[0] = 1
        self.after = NDimensionalState(state_feat)
        self.action = PrePostConditionAction(self.pre, self.post)

    def test_check_true(self):
        self.assertTrue(self.action.check(self.before, self.after))
        self.assertTrue(self.action.check(self.after, self.after))

    def test_check_false(self):
        self.assertFalse(self.action.check(self.before, self.before))
        self.assertFalse(self.action.check(self.after, self.before))

    def test_equality(self):
        mask = np.array([0, 1, 0, 0, 1])
        value = np.array([0, 0, 0, 0, -1])
        pre = Condition(mask, value)
        mask = np.array([1, 0, 0, 0, 0])
        value = np.array([1, 0, 0, 0, 0])
        post = Condition(mask, value)
        action = PrePostConditionAction(pre, post)
        self.assertEqual(self.action, action)
        self.assertEqual(hash(self.action), hash(action))

    def test_inequality(self):
        """Note that the test of hash inequality is not guaranteed to succeed.
        It should however be very unlikely to fail.
        """
        mask = np.array([0, 1, 0, 0, 1])
        value = np.array([0, 0, 0, 0, 1])
        pre = Condition(mask, value)
        action = PrePostConditionAction(pre, self.post)
        self.assertNotEqual(self.action, action)
        self.assertNotEqual(hash(self.action), hash(action))
        action = PrePostConditionAction(self.pre, self.pre)
        self.assertNotEqual(hash(self.action), hash(action))
