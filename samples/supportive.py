import os
import json

from htm.task import AbstractAction
from htm.task import (SequentialCombination, AlternativeCombination,
                      LeafCombination, ParallelCombination)
from htm.supportive import (SupportivePOMDP, AssembleFoot, AssembleTopJoint,
                            AssembleLegToTop, BringTop)
from htm.lib.pomdp import POMCPPolicyRunner


HORIZON = 100
ITERATIONS = 100
EXPLORATION = 10  # 1000
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
pol = POMCPPolicyRunner(p, iterations=ITERATIONS, horizon=HORIZON,
                        exploration=EXPLORATION,
                        relative_exploration=RELATIVE_EXPLO,
                        belief_values=BELIEF_VALUES,
                        belief='particle', belief_params={'n_particles': 100})

N = 100
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
dic = pol.trajectory_trees_from_starts()
dic['actions'] = pol.tree.model.actions
dic['states'] = pol.tree.model.states
dic['exploration'] = EXPLORATION
dic['relative_exploration'] = RELATIVE_EXPLO

with open(os.path.join(
        os.path.dirname(__file__),
        '../visualization/pomcp/json/pomcp.json'), 'w') as f:
    json.dump(dic, f, indent=2)

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
        s = ns
