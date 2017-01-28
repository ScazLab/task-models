from unittest import TestCase

import numpy as np

from htm.task_to_pomdp import (CollaborativeAction)
from htm.task import (SequentialCombination, AlternativeCombination,
                      LeafCombination, ParallelCombination)
from htm.supportive import (_HTMToDAG, unique, SupportivePOMDP, AssembleFoot,
                            AssembleTopJoint, AssembleLegToTop, BringTop,
                            CONSUMES, USES, CONSUMES_SOME,
                            _SupportivePOMDPState, NHTMHorizon)


class TestHelpers(TestCase):

    def test_unique(self):
        l = [2, 4, 1, 2, 4, 5, 1, 0]
        self.assertEqual(set(unique(l)), set([0, 1, 2, 4, 5]))


class TestHTMToDAG(TestCase):

    def setUp(self):
        a = CollaborativeAction('Do a', (3., 2., 5.))
        b = CollaborativeAction('Do b', (2., 3., 4.))
        c = CollaborativeAction('Do c', (2., 3., 4.))
        d = CollaborativeAction('Do d', (3., 2., 5.))
        self.l1 = LeafCombination(a)
        self.l2 = LeafCombination(b)
        self.l3 = LeafCombination(c)
        self.l4 = LeafCombination(d)

    def test_on_leaf(self):
        r = _HTMToDAG(self.l1)
        self.assertEqual(r.nodes, [self.l1])
        self.assertEqual(r.succs, [[]])
        self.assertEqual(r.init, [0])

    def test_on_sequence(self):
        res = _HTMToDAG(SequentialCombination([self.l1, self.l2, self.l3], name='Do all'))
        self.assertEqual(res.nodes, [self.l1, self.l2, self.l3])
        self.assertEqual(res.succs, [[1], [2], []])
        self.assertEqual(res.init, [0])

    def test_on_aternative(self):
        res = _HTMToDAG(AlternativeCombination([self.l1, self.l2, self.l3], name='Do any'))
        self.assertEqual(res.nodes, [self.l1, self.l2, self.l3])
        self.assertEqual(res.succs, [[], [], []])
        self.assertEqual(res.init, [0, 1, 2])

    def test_mixed(self):
        res = _HTMToDAG(SequentialCombination(
            [self.l1,
             AlternativeCombination([self.l2, self.l3], name='Do b or c'),
             self.l4,
             ], name='Do a b|c d'))
        self.assertEqual(res.nodes, [self.l1, self.l2, self.l3, self.l4])
        self.assertEqual(res.succs, [[1, 2], [3], [3], []])
        self.assertEqual(res.init, [0])

    def test_on_parallel(self):
        res = _HTMToDAG(ParallelCombination([self.l1, self.l2], name='Do any order'))
        self.assertEqual([n.name for n in res.nodes],
                         ['Do a order-0', 'Do b order-0', 'Do b order-1', 'Do a order-1'])
        self.assertEqual(res.succs, [[1], [], [3], []])
        self.assertEqual(res.init, [0, 2])


class TestSupportivePOMDPState(TestCase):

    def setUp(self):
        self._s = _SupportivePOMDPState(5, 3, 2, 4)

    def test_set_get_htm(self):
        self.assertEqual(self._s.htm, 0)
        for i in range(3):
            self._s.set_preference(i, 1)
        for i in range(2):
            self._s.set_body_feature(i, 1)
        for i in range(4):
            self._s.set_object(i, 1)
        self.assertEqual(self._s.htm, 0)
        self._s.htm = 3
        self.assertEqual(self._s.htm, 3)

    def test_set_get_preference(self):
        for i in range(3):
            self.assertEqual(self._s.has_preference(i), 0)
        self._s.htm = 3
        for i in range(2):
            self._s.set_body_feature(i, 1)
        for i in range(4):
            self._s.set_object(i, 1)
        for i in range(3):
            self.assertEqual(self._s.has_preference(i), 0)
        for i in range(3):
            self.assertEqual(self._s.has_preference(i), 0)
            self._s.set_preference(i, i % 2)
            self.assertEqual(self._s.has_preference(i), i % 2)

    def test_set_get_body_features(self):
        for i in range(2):
            self.assertEqual(self._s.has_body_feature(i), 0)
        self._s.htm = 3
        for i in range(3):
            self._s.set_preference(i, 1)
        for i in range(4):
            self._s.set_object(i, 1)
        for i in range(2):
            self.assertEqual(self._s.has_body_feature(i), 0)
        for i in range(2):
            self.assertEqual(self._s.has_body_feature(i), 0)
            self._s.set_body_feature(i, i % 2)
            self.assertEqual(self._s.has_body_feature(i), i % 2)

    def test_set_get_object(self):
        for i in range(4):
            self.assertEqual(self._s.has_object(i), 0)
        self._s.htm = 3
        for i in range(2):
            self._s.set_body_feature(i, 1)
        for i in range(3):
            self._s.set_preference(i, 1)
        for i in range(4):
            self.assertEqual(self._s.has_object(i), 0)
        for i in range(4):
            self.assertEqual(self._s.has_object(i), 0)
            self._s.set_object(i, i % 2)
            self.assertEqual(self._s.has_object(i), i % 2)

    def test_str(self):
        self._s.htm = 3
        for i in range(3):
            self._s.set_preference(i, i % 2)
        for i in range(2):
            self._s.set_body_feature(i, i % 2)
        for i in range(4):
            self._s.set_object(i, (i + 1) % 2)
        self.assertEqual(str(self._s), "<1701: 3 010 01 1010>")

    def test_is_final(self):
        self._s.htm = 3
        self.assertFalse(self._s.is_final())
        self._s.htm = 4
        self.assertTrue(self._s.is_final())

    def test_belief_quotient(self):
        _s = _SupportivePOMDPState(3, 1, 1, 2)
        b = np.zeros((3 * 2 ** 4))
        # bq[0] = .3
        b[_s.to_int()] = .1
        _s.set_object(0, 1)
        b[_s.to_int()] = .1
        _s.set_object(1, 1)
        b[_s.to_int()] = .1
        # bq[2] = .7
        _s.s = 0
        _s.htm = 2
        b[_s.to_int()] = .3
        _s.set_preference(0, 1)
        b[_s.to_int()] = .4
        self.assertEqual(_s.belief_quotient(b).shape, (3,))
        np.testing.assert_array_almost_equal(_s.belief_quotient(b),
                                             np.array([.3, 0., .7]))

    def test_random_object_changes(self):
        self._s.htm = 2
        self._s.set_object(1, 1)
        self._s.set_object(3, 1)
        self._s.set_preference(1, 1)
        self._s.set_body_feature(0, 1)
        self._s.random_object_changes(0.)  # p = 0 => No change
        self.assertEqual(str(self._s)[7:-1], "2 010 10 0101")
        self._s.random_object_changes(1.)  # p = 1 => XOR
        self.assertEqual(str(self._s)[7:-1], "2 010 10 1010")

    def test_random_preference_changes(self):
        self._s.htm = 2
        self._s.set_object(1, 1)
        self._s.set_object(3, 1)
        self._s.set_preference(1, 1)
        self._s.set_body_feature(0, 1)
        self._s.random_preference_changes(0.)  # p = 0 => No change
        self.assertEqual(str(self._s)[7:-1], "2 010 10 0101")
        self._s.random_preference_changes(1.)  # p = 1 => XOR
        self.assertEqual(str(self._s)[7:-1], "2 101 10 0101")


class TestSupportivePOMDP(TestCase):

    def setUp(self):
        self.bt = LeafCombination(BringTop())
        self.af = LeafCombination(AssembleFoot('leg-1'))
        self.atj = LeafCombination(AssembleTopJoint('leg-1'))
        self.alt = LeafCombination(AssembleLegToTop('leg-1'))
        self.htm = SequentialCombination([self.bt, self.af])
        self.p = SupportivePOMDP(self.htm)
        self.p.p_changed_by_human = 0.
        self.p.p_change_preference = 0.

    def test_populate_conditions(self):
        """Note: for this test we consider a requirement that objects
        are in this order but this is not a specification of the code.
        This test should be updated to something more accurate
        if the implementation changes.
        """
        self.assertEqual(self.p.objects, ['top', 'joints', 'leg', 'screwdriver', 'screws'])
        self.assertEqual(self.p.htm_conditions, [
            [(CONSUMES, 0)],
            [(USES, 1), (CONSUMES, 2),
             (USES, 3), (USES, 4)]])

    def test_last_actions_lead_to_final_state(self):
        self.assertEqual(self.p.htm_succs, [[1], [2]])

    def test_n_states(self):
        self.assertEqual(self.p.n_states, 3 * 2 ** (5 + 1 + 1))
        self.assertEqual(self.p.n_states, len(self.p.states))

    def test_features(self):
        self.assertEqual(self.p.features, [
            'HTM', 'hold-preference', 'holding', 'top', 'joints', 'leg',
            'screwdriver', 'screws'])

    def test_actions(self):
        """Same note as test_populate_conditions."""
        self.assertEqual(len(self.p.actions), self.p.n_actions)
        self.assertEqual(self.p.actions, [
            'wait', 'hold',
            'bring top',
            'bring joints', 'clear joints',
            'bring leg',
            'bring screwdriver', 'clear screwdriver',
            'bring screws', 'clear screws'])
        self.assertEqual(self.p.actions[self.p.A_WAIT], 'wait')
        self.assertEqual(self.p.actions[self.p.A_HOLD], 'hold')

    def test_action_ids(self):
        self.assertTrue(self.p._is_bring(self.p._bring(2)))
        self.assertFalse(self.p._is_bring(self.p._remove(2)))

    def test_sample_start_no_hold(self):
        self.p.p_preferences = [0]
        _s = self.p._int_to_state(self.p.sample_start())
        self.assertEqual(_s.htm, 0)
        self.assertEqual(_s.has_preference(0), 0)
        self.assertEqual(_s.has_body_feature(0), 0)
        for i in range(5):
            self.assertEqual(_s.has_object(i), 0)

    def test_sample_start_hold(self):
        self.p.p_preferences = [1.]
        _s = self.p._int_to_state(self.p.sample_start())
        self.assertEqual(_s.htm, 0)
        self.assertEqual(_s.has_preference(0), 1)
        self.assertEqual(_s.has_body_feature(0), 0)
        for i in range(5):
            self.assertEqual(_s.has_object(i), 0)

    def test_sample_transition(self):
        self.p.p_fail = 0.
        s = 0
        # Bring object
        a = self.p._bring(1)
        s, o, r = self.p.sample_transition(a, s)
        _s = self.p._int_to_state(s)
        self.assertEqual(_s.has_object(1), 1)
        self.assertEqual(o, self.p.O_NONE)
        # Remove object
        a = self.p._remove(1)
        s, o, r = self.p.sample_transition(a, s)
        _s = self.p._int_to_state(s)
        self.assertEqual(_s.has_object(1), 0)
        self.assertEqual(o, self.p.O_NONE)
        # Transition to new task state
        a = self.p.A_WAIT
        s, o, r = self.p.sample_transition(a, s)
        _s = self.p._int_to_state(s)
        self.assertEqual(_s.has_object(0), 0)  # Top not there. TODO: is it OK?
        self.assertEqual(o, self.p.O_NONE)

    def test_sample_transition_hold(self):
        htm = SequentialCombination([self.alt, self.af])
        p = SupportivePOMDP(htm)
        p.p_fail = 0.
        _s = p._int_to_state()
        _s.set_object(p.objects.index('screws'), 1)
        _s.set_object(p.objects.index('screwdriver'), 1)
        # Preference for holding
        _s.set_preference(0, 1)
        # - Wait
        s, o, r = p.sample_transition(p.A_WAIT, _s.to_int())
        self.assertEqual(r, 10.)
        # - Hold
        s, o, r = p.sample_transition(p.A_HOLD, _s.to_int())
        self.assertEqual(r, 10. - 1. + 5.)
        # No preference for holding
        _s.set_preference(0, 0)
        # - Wait
        s, o, r = p.sample_transition(p.A_WAIT, _s.to_int())
        self.assertEqual(r, 10.)
        # - Hold
        s, o, r = p.sample_transition(p.A_HOLD, _s.to_int())
        self.assertEqual(r, 10. - 1.)
        # Not required in task (Bring-top is first node in self.p)
        _s = self.p._int_to_state()
        _s.set_object(self.p.objects.index('top'), 1)
        _s.set_preference(0, 1)
        s, o, r = self.p.sample_transition(self.p.A_HOLD, _s.to_int())
        self.assertEqual(r, 10. - 1.)
        # Does not apply on final node
        _s = self.p._int_to_state()
        _s.htm = self.p.htm_final
        _s.set_preference(0, 1)
        s, o, r = self.p.sample_transition(self.p.A_HOLD, _s.to_int())
        self.assertEqual(r, - 1.)

    def test_sample_transition_action_failure(self):
        self.p.p_fail = 1.
        s = 0
        # Bring object
        a = self.p._bring(1)
        s, o, r = self.p.sample_transition(a, s)
        _s = self.p._int_to_state(s)
        self.assertEqual(_s.has_object(1), 0)
        self.assertEqual(o, self.p.O_FAIL)
        # Remove object
        _s = self.p._int_to_state()
        _s.set_object(1, 1)
        a = self.p._remove(1)
        s, o, r = self.p.sample_transition(a, _s.to_int())
        _s = self.p._int_to_state(s)
        self.assertEqual(_s.has_object(1), 1)
        self.assertEqual(o, self.p.O_FAIL)

    def test_sample_transition_not_found(self):
        s = 0
        # Remove object
        a = self.p._remove(1)
        s, o, r = self.p.sample_transition(a, s)
        _s = self.p._int_to_state(s)
        self.assertEqual(_s.has_object(1), 0)
        self.assertEqual(o, self.p.O_NOT_FOUND)
        # Remove object
        _s = self.p._int_to_state()
        _s.set_object(1, 1)
        a = self.p._bring(1)
        s, o, r = self.p.sample_transition(a, _s.to_int())
        _s = self.p._int_to_state(s)
        self.assertEqual(_s.has_object(1), 1)
        self.assertEqual(o, self.p.O_NOT_FOUND)


class TestNHTMHorizon(TestCase):

    def setUp(self):
        self.bt = LeafCombination(BringTop())
        self.af = LeafCombination(AssembleFoot('leg-1'))
        self.htm = SequentialCombination([self.bt, self.af])
        self.model = SupportivePOMDP(self.htm)
        self.h = NHTMHorizon(self.model, 1)

    def is_not_reached(self):
        self.assertFalse(self.h.is_reached())

    def test_no_htm_transition(self):
        _s = self.model._int_to_state()
        _new_s = self.model._int_to_state()
        _new_s.set_object(2, 1)
        _new_s.set_preference(0, 1)
        self.h.decrement(1, _s.to_int(), _new_s.to_int(), 0)
        self.assertFalse(self.h.is_reached())

    def test_htm_transition(self):
        _s = self.model._int_to_state()
        _new_s = self.model._int_to_state()
        _new_s.htm = 1
        self.h.decrement(1, _s.to_int(), _new_s.to_int(), 0)
        self.assertTrue(self.h.is_reached())

    def test_reached_on_final(self):
        _s = self.model._int_to_state()
        _s.htm = 2
        s = _s.to_int()
        self.h.decrement(1, s, s, 0)
        self.assertTrue(self.h.is_reached())
