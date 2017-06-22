import time
import math
import json
import threading
from numbers import Integral

import numpy as np

from .belief import (ArrayBelief, ParticleBelief, MaxSamplesReached,
                     format_belief_array)
from .multiprocess import repeat


class Horizon(object):

    class _Generator:

        def __init__(self, cls, *args, **kwargs):
            self.cls = cls
            self.args = args
            self.kwargs = kwargs

        def __call__(self):
            return self.cls(*self.args, **self.kwargs)

    def is_reached(self):
        raise NotImplementedError

    def decrement(self, a, s, new_s, o):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    @classmethod
    def generator(cls, model, **parameters):
        raise NotImplementedError


class NTransitionsHorizon(Horizon):

    def __init__(self, n):
        self.n = n

    def is_reached(self):
        return self.n <= 0

    def decrement(self, a, s, new_s, o):
        self.n -= 1

    def copy(self):
        return NTransitionsHorizon(self.n)

    @classmethod
    def generator(cls, model, n=100):
        return cls._Generator(cls, n)


def _children_to_dict(d, children):
    d['children'] = children
    return d


def _null_logger(*args, **kwargs):
    pass


class _SearchTree(object):

    rollout_it = 100

    def __init__(self, model, horizon_generator, exploration,
                 relative_exploration=False, belief='array', belief_params={},
                 node_params={}, logger=None, multiprocess=False):
        self._belief = belief
        self._belief_params = belief_params
        self.model = model
        self._node_params = node_params
        self.root = self._observation_node_for_belief(self._belief_start())
        self.horizon_gen = horizon_generator
        self.exploration = exploration
        self.relative_explo = relative_exploration
        self.log = _null_logger if logger is None else logger
        self.multiprocess = multiprocess

    def _belief_start(self):
        if self._belief == 'array':
            return ArrayBelief(self.model.start, **self._belief_params)
        elif self._belief == 'particle':
            return ParticleBelief(self.model.sample_start, self.model.n_states,
                                  **self._belief_params)
        else:
            raise ValueError('Unknown belief type: ' + self._belief)

    def get_node(self, history):
        """Raises ValueError if node does not exist or history is invalid."""
        node = self.root
        last_belief = node.belief
        for i, h in enumerate(history):
            if isinstance(node, _SearchActionNode):  # h is an observation
                if h not in node.children:
                    node.children[h] = self._observation_node_for_belief(
                        last_belief.successor(self.model, history[i - 1], h))
                node = node.children[h]
            else:  # h is an action
                last_belief = node.belief
                node = node.safe_get_child(h)
        return node

    def random_action(self):
        return np.random.randint(self.model.n_actions)

    def rollout_from_node(self, node, horizon):
        if horizon.is_reached():
            return 0
        else:

            def one_rollout():
                return self._one_rollout_from_belief(node.belief,
                                                     horizon.copy())

            if self.multiprocess:
                returns = repeat(one_rollout, self.rollout_it)
            else:
                returns = [one_rollout() for _ in range(self.rollout_it)]
            avg_return = sum(returns) / self.rollout_it  # Avg over rollouts
            node.update(avg_return)  # Only counts one visit
            return avg_return

    def _one_rollout_from_belief(self, belief, horizon):
        return self._one_rollout_from_state(belief.sample(), horizon)

    def _one_rollout_from_state(self, state, horizon):
        gamma = 1.
        full_return = 0.
        while not horizon.is_reached():
            a = self.random_action()
            new_state, o, r = self.model.sample_transition(a, state)
            horizon.decrement(a, state, new_state, o)
            state = new_state
            full_return += gamma * r
            gamma *= self.model.discount
        return full_return

    def simulate_from_node(self, node, action=None):
        state = node.belief.sample()
        self._simulate_from_node(node, state, self.horizon_gen(), a=action)

    def _observation_node_for_belief(self, b):
        return _SearchObservationNode(b, self.model.n_actions, **self._node_params)

    def _simulate_from_node(self, node, state, horizon, a=None):
        if horizon.is_reached():
            return node.value
        else:
            if a is None:
                a = node.get_best_action(
                    exploration=self.exploration,
                    relative_exploration=self.relative_explo)
            child = node.safe_get_child(a)
            new_s, o, r = self.model.sample_transition(a, state)
            horizon.decrement(a, state, new_s, o)
            if o not in child.children:
                try:
                    # Create node with updated belief
                    child.children[o] = self._observation_node_for_belief(
                        node.belief.successor(self.model, a, o))
                    # Use rollout
                    partial_return = self.rollout_from_node(child.children[o],
                                                            horizon)
                except MaxSamplesReached:
                    self.log('Maximum number of samples reached, skipping.')
                    partial_return = 0.
                    # Note maybe use more relevant value, but, since the event
                    # is rare, it should not impact the result
            else:
                # Continue regular search
                partial_return = self._simulate_from_node(
                    child.children[o], new_s, horizon)
            full_return = r + self.model.discount * partial_return
            child.update(full_return)
            node.update(full_return)
            # TODO belief update (not needed for exact belief)
            return full_return

    def to_dict(self, as_policy=False):
        return self.root.to_dict(self.model, as_policy=as_policy)

    def map(self, fun, join_children=_children_to_dict):
        """
        :param join_children: function(result, children)
            returns result_with_children
        """
        return self.root._map(fun, join_children)


class _ObservationLookupSearchTree(_SearchTree):

    def __init__(self, model, horizon, exploration,
                 relative_exploration=False, belief='array', belief_params={},
                 node_params={}, logger=None, multiprocess=False):
        self._obs_nodes = {}  # used in super for root initialization
        if belief == 'particle':
            raise ValueError(
                '_ObservationLookupSearchTree does not support particle belief')
        super(_ObservationLookupSearchTree, self).__init__(
            model, horizon, exploration,
            relative_exploration=relative_exploration,
            belief=belief, belief_params={},
            node_params=node_params, logger=logger, multiprocess=multiprocess)

    def _observation_node_for_belief(self, b):
        # Returns node for given belief, creating one if none exists
        if b not in self._obs_nodes:
            self._obs_nodes[b] = _SearchObservationNode(
                b, self.model.n_actions, **self._node_params)
        return self._obs_nodes[b]

    # Here we need to keep track of visited children since the tree is no more
    # a tree...
    def to_dict(self, as_policy=False):
        return self.root.to_dict(self.model, as_policy=as_policy,
                                 exclude_visited=set())


class _ValueAverage(object):

    def __init__(self, alpha=0):
        self.n_simulations = 0
        self.total_value = 0.
        assert(0 <= alpha <= 1)
        self.alpha = alpha

    @property
    def value(self):
        return 0. if self.n_simulations == 0 \
            else self.total_value / self.n_simulations

    def update(self, value):
        self.total_value = ((self.total_value + value) * (1 - self.alpha) +
                            self.alpha * (self.n_simulations + 1) * value)
        self.n_simulations += 1


class _SearchNode(object):

    def __init__(self, alpha=.001):
        self._avg = _ValueAverage(alpha=alpha)
        self.children = {}

    def __str__(self):
        return "[" + ", ".join(["{}: {}".format(i, self.children[i])
                                for i in self._children_keys()]) + "]"

    def _children_keys(self):
        return sorted(self.children.keys())

    @property
    def n_simulations(self):
        return self._avg.n_simulations

    @property
    def value(self):
        return self._avg.value

    def update(self, value):
        self._avg.update(value)

    def to_dict(self, model, as_policy=False, exclude_visited=None,
                recursive=True):
        return {"value": self.value,
                "visits": self.n_simulations,
                "node": None,
                }

    def _map(self, fun, join_children):
        result = fun(self)
        child_results = [c._map(fun, join_children)
                         for c in self._iterate_children()]
        return join_children(result, child_results)

    def _iterate_children(self):
        return self.children.values()


class _SearchObservationNode(_SearchNode):
    """
    Children indexed by action.
    """

    def __init__(self, belief, n_actions, alpha=.001):
        super(_SearchObservationNode, self).__init__(alpha=alpha)
        self.belief = belief
        self.children = [None for _ in range(n_actions)]
        self._children_alpha = alpha

    def children_dict(self, model):
        return {model.actions[a]: c
                for a, c in enumerate(self.children) if c is not None}

    def _children_keys(self):
        return [i for i, c in enumerate(self.children) if c is not None]

    def _not_init_children(self):
        return [i for i, c in enumerate(self.children)
                if c is None or c.n_simulations == 0]

    def augmented_values(self, exploration=0, relative=False):
        # Note: nans are returned for not initialized children
        if exploration > 0 and relative:
            vals = [child.value if child is not None else np.nan
                    for child in self.children]
            exploration *= np.nanmax(vals) - np.nanmin(vals)
        l_ns = np.log(self.n_simulations)
        return [child.value + exploration * np.sqrt(l_ns / child.n_simulations)
                if child is not None else np.nan
                for child in self.children]

    def get_best_action(self, exploration=0, relative_exploration=False):
        not_init = self._not_init_children()
        if len(not_init) == 0:
            assert(self.n_simulations > 0)  # explored if children explored
            # Augmented greedy (UCT)
            a = np.argmax([self.augmented_values(
                exploration=exploration, relative=relative_exploration)])
        else:
            # Chose an unexplored action
            a = np.random.choice(not_init)
        return a

    def safe_get_child(self, a):
        if self.children[a] is None:
            self.children[a] = _SearchActionNode(alpha=self._children_alpha)
        return self.children[a]

    def _iterate_children(self):
        return filter(lambda c: c is not None, self.children)

    def to_dict(self, model, as_policy=False, observed=None,
                exclude_visited=None, recursive=True):
        children = recursive
        if exclude_visited is not None:
            if self.belief in exclude_visited:
                children = False
            else:
                exclude_visited.add(self.belief)
        base = super(_SearchObservationNode, self).to_dict(
            model, as_policy=as_policy, exclude_visited=exclude_visited)
        base["belief"] = self.belief.to_list()
        if as_policy:
            a = self.get_best_action()
            grand_children = self.safe_get_child(a).children
            base.update({
                "action": model.actions[a],
                "observed": observed,
                "values": [v if not math.isnan(v) else None
                           for v in self.augmented_values()],  # For json
                "exploration_terms": [
                    np.sqrt(np.log(self.n_simulations) / child.n_simulations)
                    if ((child is not None) and child.n_simulations > 0)
                    else None
                    for child in self.children
                    ],
                "child_visits": [c.n_simulations if c is not None else 0
                                 for c in self.children],
                })
            if children:
                base.update({
                    "observations": [model.observations[o]
                                     for o in grand_children],
                    "children": [
                        grand_children[o].to_dict(
                            model, as_policy=as_policy, observed=i,
                            exclude_visited=exclude_visited)
                        for i, o in enumerate(grand_children)],
                    })
        else:
            if children:
                base.update({
                    "actions": [model.actions[i]
                                for i, c in enumerate(self.children)
                                if c is not None],
                    "children": [c.to_dict(model, as_policy=as_policy,
                                           exclude_visited=exclude_visited)
                                 for c in self.children if c is not None],
                    })
            else:
                base.update({'actions': [], 'children': []})

        return base


class _SearchActionNode(_SearchNode):
    """
    Children indexed by observation.
    """

    def to_dict(self, model, as_policy=False, exclude_visited=None,
                recursive=True):
        if as_policy:
            raise NotImplemented
        else:
            base = super(_SearchActionNode, self).to_dict(
                model, as_policy=as_policy, exclude_visited=exclude_visited)
            base.update({
                "observations": [model.observations[o]
                                 for o in self.children],
                "children": [self.children[o].to_dict(
                                model, as_policy=as_policy,
                                exclude_visited=exclude_visited)
                             for o in self.children],
                })
            return base


class POMCPPolicyRunner(object):
    """
    :param particles: number of particles for belief estimation
    :param horizon: length of simulation episodes
    :param iterations: number of simulation episodes to run
    :param exploration: UCT exploration parameter (c in [Silver2010])
    :param belief_values: group values for histories with same belief
    """

    def __init__(self, model, particles=20, iterations=100, horizon=100,
                 exploration=None, relative_exploration=False,
                 belief_values=False, belief='array', belief_params={},
                 logger=None, multiprocess_rollouts=True):
        if logger is None:
            from logging import warning as logger
        if exploration is None:
            exploration = 1. if relative_exploration else 100
        tree_class = (_ObservationLookupSearchTree if belief_values
                      else _SearchTree)
        if isinstance(horizon, Horizon._Generator):
            horizon_generator = horizon
        elif isinstance(horizon, Integral):
            horizon_generator = NTransitionsHorizon.generator(model, n=horizon)
        else:
            raise ValueError('Invalid horizon: ' + str(horizon))
        self.tree = tree_class(model, horizon_generator, exploration,
                               relative_exploration=relative_exploration,
                               belief=belief, belief_params=belief_params,
                               logger=logger,
                               multiprocess=multiprocess_rollouts)
        self.iterations = iterations
        self._reset()

    @property
    def actions(self):
        return self.tree.model.actions

    @property
    def observations(self):
        return self.tree.model.observations

    def reset(self, belief=None):
        self._reset(belief=belief)

    # Note the reason for having both _reset and reset
    # is that in the async subclass the former needs the thread
    # to be initialized, and __init__ calls _reset before the that.
    def _reset(self, belief=None):
        if belief is not None:
            raise NotImplementedError
        self.history = []
        self._node = self.tree.root
        self._last_action = None

    @property
    def belief(self):
        return self._node.belief

    def get_action(self, iterations=None):
        # Note iterations must be greater than the number of actions
        # to guarantee that any action chosen as best_action is explored first
        if iterations is None:
            iterations = self.iterations
        for _ in range(iterations):
            self.tree.simulate_from_node(self._node)
        a = self._node.get_best_action()
        # No exploration during exploitation?
        self._last_action = a
        return self.actions[a]

    def step(self, observation):
        if self._last_action is None:
            raise ValueError('Unknown last action')
            # TODO rethink the design of the PolicyRunner class
        o = self.observations.index(observation)
        self.history.extend([self._last_action, o])
        # TODO: Make sure that node exists (and explores from previous)
        self._node = self.tree.get_node(self.history)

    def trajectory_trees_from_starts(self, qvalue=False):
        return {"graphs": [self.tree.to_dict(as_policy=not qvalue)]}

    def run_trajectory(self, logger=None):
        if logger is None:
            from logging import info as logger
        model = self.tree.model
        R = 0
        logger('New trajectory')
        self.reset()
        s = model.sample_start()
        belief_quotient = model._int_to_state().belief_quotient
        belief_preferences = model._int_to_state().belief_preferences
        while not model.is_final(s):
            a = self.get_action()
            ns, o, r = model.sample_transition(model.actions.index(a), s)
            self.step(model.observations[o])
            logger('{} -- {} --> {}, {}, {}'.format(
                model._int_to_state(s),
                a,
                model._int_to_state(ns),
                model.observations[o],
                r))
            logger('belief: {} | {:.2f}'.format(
                format_belief_array(belief_quotient(self.belief.array)),
                belief_preferences(self.belief.array)[0]))
            s = ns
            R += r
        logger("Total reward: %f" % R)


class AsyncPOMCPPolicyRunner(POMCPPolicyRunner):

    class _Thread(threading.Thread):
        """Continuously explores and enable another thread to execute
        some exploitation in between two explorations.
        """

        def __init__(self, tree):
            super(AsyncPOMCPPolicyRunner._Thread, self).__init__()
            self.tree = tree
            self._node = tree.root
            self._action = None
            self._done = False
            self._lock = threading.Lock()

        def _stop(self):
            self._done = True

        def stop(self):
            self.execute(self._stop)

        def execute(self, fun, *args, **kwargs):
            """Waits until current exploration is done and execute fun.
            """
            with self._lock:
                return fun(*args, **kwargs)

        def set_node(self, node):
            self._node = node
            self._action = None

        def set_action(self, action):
            self._action = action

        def explore(self):
            self.tree.simulate_from_node(self._node, action=self._action)

        def run(self):
            done = False
            while not done:
                with self._lock:
                    self.explore()
                    done = self._done
                time.sleep(.01)

    def __init__(self, *args, **kwargs):
        super(AsyncPOMCPPolicyRunner, self).__init__(*args, **kwargs)
        self.thread = self._Thread(self.tree)
        self.thread.start()

    def step(self, observation):
        super(AsyncPOMCPPolicyRunner, self).step(observation)
        self.thread.execute(self.thread.set_node, self._node)

    def reset(self, belief=None):
        self._reset(belief=belief)
        self.thread.execute(self.thread.set_node, self._node)

    def get_action(self, iterations=None):
        # Start with minimal number of iterations to have explored all actions
        a = self.thread.execute(super(AsyncPOMCPPolicyRunner, self).get_action)
        self.thread.execute(self.thread.set_action, self._last_action)
        return a

    def stop(self):
        self.thread.stop()
        if self.thread.isAlive():
            self.thread.join()

    def execute(self, *args, **kwargs):
        self.thread.execute(*args, **kwargs)

    def __del__(self):
        self.stop()


def export_pomcp(policy, destination, belief_as_quotient=False):
    model = policy.tree.model
    if belief_as_quotient:
        dic = {}
        FLAG = '####'

        def to_dict(node):
            if isinstance(node, _SearchObservationNode):
                d = node.to_dict(model, as_policy=True, recursive=False)
                d['belief'] = list(model._int_to_state().belief_quotient(
                    np.array(d['belief'])))
                a_i = model.actions.index(d['action'])
                d['ACTION_IDX'] = sum([c is not None
                                       for c in node.children[:a_i]])
            else:
                d = {"observations": [model.observations[o]
                                      for o in node.children]}
                d[FLAG] = True
            return d

        def join_children(d, children):
            if d.get(FLAG, False):  # Action node
                d.pop(FLAG)
                d['children'] = children
            else:  # Observation node
                i = d.pop('ACTION_IDX')
                child = children[i]
                d['observations'] = child['observations']
                d['children'] = child['children']
                for c, o in zip(d['children'], d['observations']):
                    c['observed'] = model.observations.index(o)
            return d

        dic['graphs'] = [policy.tree.map(to_dict, join_children)]
        dic['states'] = model.htm_names
    else:
        dic = policy.trajectory_trees_from_starts()
        dic['states'] = model.states
    dic['actions'] = model.actions
    dic['exploration'] = policy.tree.exploration
    dic['relative_exploration'] = policy.tree.relative_explo

    with open(destination, 'w') as f:
        json.dump(dic, f, indent=2)
