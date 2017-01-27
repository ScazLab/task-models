import os
import json

import numpy as np

from htm.task import AbstractAction
from htm.task import (SequentialCombination, AlternativeCombination,
                      LeafCombination, ParallelCombination)
from htm.supportive import (SupportivePOMDP, AssembleFoot, AssembleTopJoint,
                            AssembleLegToTop, BringTop, NHTMHorizon)
from htm.lib.pomdp import POMCPPolicyRunner


def _format_p(x):
    s = "{:0.1f}".format(x)
    return "1." if s == "1.0" else s[1:]


def format_belief(b):  # b is an array
    return " ".join([_format_p(p) for p in b])


N = 100  # for warm-up
ITERATIONS = 200
EXPLORATION = 10  # 1000
N_PARTICLES = 200
RELATIVE_EXPLO = True  # In this case use smaller exploration
BELIEF_VALUES = False


leg_i = 'leg-{}'.format
mount_legs = SequentialCombination([
    SequentialCombination([LeafCombination(AssembleFoot(leg_i(i))),
                           LeafCombination(AssembleTopJoint(leg_i(i))),
                           LeafCombination(AssembleLegToTop(leg_i(i))),
                           ])
    for i in range(4)])
htm = SequentialCombination([LeafCombination(BringTop()), mount_legs])

p = SupportivePOMDP(htm)
pol = POMCPPolicyRunner(p, iterations=ITERATIONS,
                        horizon=NHTMHorizon.generator(p, n=3),
                        exploration=EXPLORATION,
                        relative_exploration=RELATIVE_EXPLO,
                        belief_values=BELIEF_VALUES,
                        belief='particle',
                        belief_params={'n_particles': N_PARTICLES})

def export_policy():
    EXPORT_BELIEF_QUOTIENT = True
    best = None
    maxl = 0
    for i in range(N):
        s = 'Exploring... [{:2.0f}%] (current best: {} [{:.1f}])'.format(
                i * 100. / N, best, pol.tree.root.children[pol._last_action].value
                if pol._last_action is not None else 0.0)
        maxl = max(maxl, len(s))
        print(' ' * maxl, end='\r')
        print(s, end='\r')
        best = pol.get_action()  # Some exploration
    print('Exploring... [done]')
    if BELIEF_VALUES:
        print('Found {} distinct beliefs.'.format(len(pol.tree._obs_nodes)))
    if EXPORT_BELIEF_QUOTIENT:
        from htm.lib.pomdp import _SearchObservationNode
        dic = {}
        FLAG = '###'

        def to_dict(node):
            if isinstance(node, _SearchObservationNode):
                d = node.to_dict(pol.tree.model, as_policy=True, recursive=False)
                d['belief'] = list(pol.tree.model._int_to_state().belief_quotient(np.array(d['belief'])))
                a_i = pol.tree.model.actions.index(d['action'])
                d['ACTION_IDX'] = sum([c is not None for c in node.children[:a_i]])
            else:
                d = {"observations": [pol.tree.model.observations[o]
                                      for o in node.children]}
                d[FLAG] = True
            return d

        def join_children(d, children):
            if d.get(FLAG, False):  # Action node
                d.pop(FLAG)
                d['children'] = children
            else:
                i = d.pop('ACTION_IDX')
                child = children[i]
                d['observations'] = child['observations']
                d['children'] = child['children']
                for c, o in zip(d['children'], d['observations']):
                    c['observed'] = pol.tree.model.observations.index(o)
            return d

        dic['graphs'] = [pol.tree.map(to_dict, join_children)]
        dic['states'] = [n.name for n in pol.tree.model.htm_nodes] + ['final']
    else:
        dic = pol.trajectory_trees_from_starts()
        dic['states'] = pol.tree.model.states
    dic['actions'] = pol.tree.model.actions
    dic['exploration'] = EXPLORATION
    dic['relative_exploration'] = RELATIVE_EXPLO

    with open(os.path.join(
            os.path.dirname(__file__),
            '../visualization/pomcp/json/pomcp.json'), 'w') as f:
        json.dump(dic, f, indent=2)


export_policy()

# Play trajectories
for _ in range(5):
    print('New trajectory')
    pol.reset()
    s = p.sample_start()
    while not p.is_final(s):
        a = pol.get_action()
        ns, o, r = p.sample_transition(p.actions.index(a), s)
        pol.step(p.observations[o])
        print('{} -- {} --> {}, {}, {}'.format(
            p._int_to_state(s), a, p._int_to_state(ns), p.observations[o], r))
        print('belief: ',
              format_belief(pol.tree.model._int_to_state().belief_quotient(pol.belief.array)))
        s = ns

export_policy()
