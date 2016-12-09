from __future__ import unicode_literals
import os

from htm.task_to_pomdp import HTMToPOMDP
from htm.stool_scenarios import (stool_task_sequential, T_WAIT, T_ASK, T_TELL,
                                 C_INTR)


## Tested scenarios:
# 1. with full sequence of sequential actions
R_END = 0.1
LOOP = False
# 2. with full sequence of sequential actions
# R_END = 100
# LOOP = True

## Convert the task into a POMDP

h2p = HTMToPOMDP(T_WAIT, T_ASK, T_TELL, C_INTR, end_reward=R_END, loop=LOOP)
p = h2p.task_to_pomdp(stool_task_sequential)
#p.discount = .99

from htm.lib.pomdp import GraphPolicyBeliefRunner, POMCPPolicyRunner

pol = POMCPPolicyRunner(p, iterations=10000, horizon=30)
pol.get_action()  # Some exploration
print(pol.trajectory_trees_from_starts(qvalue=True))
dic = pol.trajectory_trees_from_starts()
import json
with open(os.path.join(
    os.path.dirname(__file__),
    '../visualization/trajectories/json/trajectories2.json'), 'w') as f:
    json.dump(dic, f, indent=2)


def pomcp_pseudo_value(pol, breadth_first=True, max_nodes=None):
    model = pol.tree.model
    histories = []
    nodes = []
    heap = [([], pol.tree.root)]
    while len(heap) > 0 and max_nodes is None or len(nodes) < max_nodes:
        h, n = heap.pop(0 if breadth_first else -1)
        histories.append(h)
        nodes.append(n)
        d = n.children_dict(model)
        for a in d:
            for o in d[a].children:
                obs = model.observations[o]
                heap.append((h + [a, obs], d[a].children[o]))
    print('Found {} nodes'.format(len(nodes)))
    return histories, nodes


def by_two(l):
    # Note: silently ignores last if number of elements is not pair
    it = l.__iter__()
    while True:  # will raise StopIteration on empty
        yield it.__next__(), it.__next__()


def short_a(action):
    return "".join([s[0] for s in action.split('-')])


def short_o(observation):
    if observation == 'none':
        return '∅'
    else:
        return observation[0]


def format_history(histories):
    return ", ".join(["{}:{}".format(short_a(a), short_o(o))
                      for a, o in by_two(histories)])


from htm.plot import plot_beliefs
import matplotlib.pyplot as plt
import numpy as np


plt.interactive(True)
plt.close('all')

model = pol.tree.model
histories, nodes = pomcp_pseudo_value(pol, max_nodes=50)

fig, axs = plt.subplots(1, 2, sharey=True)
b = np.vstack([n.belief.array for n in nodes])
plot = plot_beliefs(b, states=model.states, xlabels_rotation=45, ax=axs[0])
plt.colorbar(plot)
plt.title('Beliefs')

q = np.array([[c.value if c is not None else 0. for c in n.children]
              for n in nodes])
plot = plot_beliefs(q, states=model.actions, xlabels_rotation=45,
                    ylabels=[format_history(h) for h in histories], ax=axs[1])
plt.colorbar(plot)
plt.title('Values')
