"""This module provides classes to represent actions that can only account for
specific state transitions. These action objects implement a `check` method
that informs whether a transition is plausible for a given action.

The module implements one flavor of such actions (and associated conditions)
for which the check is true when a pre-condition and a post-condition are
simultaneously satisfied.
"""


import numpy as np


class Condition(object):
    """Pre or post condition on an environment state.

    Defined by a mask and feature values.
    """

    def __init__(self, mask, value):
        if np.count_nonzero((1 - mask) * value) != 0:
            raise ValueError('Masked values must be zeroed.')
        self.mask = mask
        self.value = value

    def __hash__(self):
        return hash((self.mask.tostring(), self.value.tostring()))

    def __eq__(self, other):
        return (isinstance(other, Condition) and
                np.array_equal(self.mask, other.mask) and
                np.array_equal(self.value, other.value))

    def check(self, state):
        return (state.get_features() * self.mask == self.value).all()


class MatchAllCondition(Condition):
    """Always true condition.
    """

    def __init__(self):
        pass

    def __hash__(self):
        return hash(0)

    def __eq__(self, other):
        return isinstance(other, MatchAllCondition)

    def check(self, state):
        return True


class Action(object):
    """Base class for actions that provide a check method."""

    def __init__(self, name="unnamed-action", agent = "robot"):
        self.name = name
        self.agent = agent

    def __repr__(self):
        return "{}<{}>".format(self.__class__.__name__, self.name)

    def __str__(self):
        return self.name

    def check(self, before, after):
        """Checks whether the action can explain a transition from the before
        and after states.
        """
        raise NotImplementedError


class PrePostConditionAction(Action):
    """Action which can apply to states verifying the pre-condition and can
    produce states that verify the post-condition.
    """

    def __init__(self, pre_condition, post_condition, name="unnamed-action"):
        super(PrePostConditionAction, self).__init__(name=name)
        self.pre = pre_condition
        self.post = post_condition

    def __hash__(self):
        return hash((self.pre, self.post))

    def __eq__(self, other):
        return (isinstance(other, Action) and
                self.pre == other.pre and
                self.post == other.post)

    def check(self, before, after):
        return self.pre.check(before) and self.post.check(after)
