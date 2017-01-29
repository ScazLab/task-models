from __future__ import print_function

import os

from htm.task import SequentialCombination, LeafCombination
from htm.supportive import (SupportivePOMDP, AssembleLeg, AssembleLegToTop,
                            NHTMHorizon)
from htm.lib.pomdp import POMCPPolicyRunner, export_pomcp
from htm.lib.belief import format_belief_array


N = 100  # for warm-up
ITERATIONS = 200
EXPLORATION = 10  # 1000
N_PARTICLES = 200
RELATIVE_EXPLO = True  # In this case use smaller exploration
BELIEF_VALUES = False
EXPORT_BELIEF_QUOTIENT = True
POMCP_DESTINATION = os.path.join(os.path.dirname(__file__),
                                 '../visualization/pomcp/json/pomcp.json')
HORIZON = 4


leg_i = 'leg-{}'.format
htm = SequentialCombination([
    SequentialCombination([
        LeafCombination(AssembleLeg(leg_i(i))),
        LeafCombination(AssembleLegToTop(leg_i(i), bring_top=(i == 0)))])
    for i in range(4)])

p = SupportivePOMDP(htm)
pol = POMCPPolicyRunner(p, iterations=ITERATIONS,
                        horizon=NHTMHorizon.generator(p, n=HORIZON),
                        exploration=EXPLORATION,
                        relative_exploration=RELATIVE_EXPLO,
                        belief_values=BELIEF_VALUES,
                        belief='particle',
                        belief_params={'n_particles': N_PARTICLES})


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


export_pomcp(pol, POMCP_DESTINATION, belief_as_quotient=EXPORT_BELIEF_QUOTIENT)

# Play trajectories
for _ in range(5):
    R = 0
    print('New trajectory')
    pol.reset()
    s = p.sample_start()
    belief_quotient = pol.tree.model._int_to_state().belief_quotient
    belief_preferences = pol.tree.model._int_to_state().belief_preferences
    while not p.is_final(s):
        a = pol.get_action()
        ns, o, r = p.sample_transition(p.actions.index(a), s)
        pol.step(p.observations[o])
        print('{} -- {} --> {}, {}, {}'.format(
            p._int_to_state(s), a, p._int_to_state(ns), p.observations[o], r))
        print('belief: ',
              format_belief_array(belief_quotient(pol.belief.array)),
              ' | {:.2f}'.format(belief_preferences(pol.belief.array)[0]),
              )
        s = ns
        R += r
    print("Total reward: %f" % R)

export_pomcp(pol, POMCP_DESTINATION, belief_as_quotient=EXPORT_BELIEF_QUOTIENT)
