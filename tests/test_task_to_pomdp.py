from unittest import TestCase

from htm.task import HierarchicalTask, LeafCombination
from htm.task_to_pomdp import HTM2POMDP, CollaborativeAction


class TestHTM2POMDP(TestCase):

    def setUp(self):
        self.h2p = HTM2POMDP(1., 2.)

    def test_name_radix(self):
        a = CollaborativeAction('My mixed Case action', (3., 2., 5.))
        self.assertEqual(self.h2p._name_radix(a), 'my-mixed-case-action')

    def test_leaf_to_pomdp(self):
        task = HierarchicalTask(root=LeafCombination(
            CollaborativeAction('Do it', (3., 2., 5.))))
        self.h2p.task_to_pomdp(task)
        # TODO: some actual tests
