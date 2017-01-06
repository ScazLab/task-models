from __future__ import unicode_literals
import os
import json
from heapq import heappush, heappop

import matplotlib.pyplot as plt
import numpy as np

from htm.lib.pomdp import GraphPolicyBeliefRunner, POMCPPolicyRunner
from htm.task_to_pomdp import HTMToPOMDP
from htm.stool_scenarios import (stool_task_sequential, T_WAIT, T_ASK, T_TELL,
                                 C_INTR)
from htm.plot import plot_beliefs


def push_last(queue, value, x):
    queue.append(x)


def push_heap(queue, value, x):
    heappush(queue, (-value, x))  # We want to access largest values first


def pop_first(queue):
    return queue.pop(0)


def pop_last(queue):
    return queue.pop()


def pop_heap(queue):
    _, x = heappop(queue)
    return x


def node_search(pol, order='q', max_nodes=None):
    heap = []
    if order == 'breadth':
        push, pop = push_last, pop_first
    elif order == 'depth':
        push, pop = push_last, pop_last
    elif order == 'q':  # best actions explored first
        push, pop = push_heap, pop_heap
    else:
        raise ValueError('Unknown order ' + str(order))
    model = pol.tree.model
    histories = []
    nodes = []
    push(heap, pol.tree.root.value, ([], pol.tree.root))
    # Note: The value is not actually useful for the root
    while len(heap) > 0 and (max_nodes is None or len(nodes) < max_nodes):
        h, n = pop(heap)
        histories.append(h)
        nodes.append(n)
        d = n.children_dict(model)
        for a in d:
            for o in d[a].children:
                obs = model.observations[o]
                child = d[a].children[o]
                push(heap, child.value, (h + [a, obs], child))
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


def format_history(history):
    return "{} [{}]".format(
        ", ".join(["{}:{}".format(short_a(a), short_o(o))
                   for a, o in by_two(history)]),
        len(history))


## Tested scenarios:
# 1. with full sequence of sequential actions
R_END = 500.
LOOP = False
R_STATE = True
# 2. with full sequence of sequential actions
# R_END = 100
# LOOP = True

HORIZON = 100
ITERATIONS = 1000
EXPLORATION = 50.

## Convert the task into a POMDP

h2p = HTMToPOMDP(T_WAIT, T_ASK, T_TELL, C_INTR, end_reward=R_END, loop=LOOP,
                 reward_state=R_STATE)
p = h2p.task_to_pomdp(stool_task_sequential)
#p.discount = .99

pol = POMCPPolicyRunner(p, iterations=ITERATIONS, horizon=HORIZON,
                        exploration=EXPLORATION)
N = 10
best = None
maxl = 0
for i in range(N):
    s = 'Exploring... [{:.0f}%] (current best: {} [{:.1f}])'.format(
            i * 100. / N, best, pol.tree.root.children[pol._last_action].value
            if pol._last_action is not None else 0.0)
    maxl = max(maxl, len(s))
    print(' ' * maxl, end='\r')
    print(s, end='\r')
    best = pol.get_action()  # Some exploration
print('Exploring... [done]')
dic = pol.trajectory_trees_from_starts()
dic['actions'] = pol.tree.model.actions
dic['states'] = pol.tree.model.states

with open(os.path.join(
        os.path.dirname(__file__),
        '../visualization/pomcp/json/pomcp.json'), 'w') as f:
    json.dump(dic, f, indent=2)


plt.interactive(True)
plt.close('all')

model = pol.tree.model
#model.R += 101.  # Make all rewards positive
histories, nodes = node_search(pol, order='q', max_nodes=50)

fig, axs = plt.subplots(1, 3, sharey=True)
b = np.vstack([n.belief.array for n in nodes])
plot = plot_beliefs(b, states=model.states, xlabels_rotation=45, ax=axs[0])
plt.colorbar(plot, ax=axs[0])
plt.title('Beliefs')

q = np.array([[c.value if c is not None else 0. for c in n.children]
              for n in nodes])
plot = plot_beliefs(q, states=model.actions, xlabels_rotation=45,
                    ax=axs[1])
plt.colorbar(plot, ax=axs[1])
axs[1].set_title('Values')
aq = np.array([
    n.augmented_values(pol.tree.exploration)
        if len(n._not_init_children()) == 0
        else [0. for _ in range(model.n_actions)]
    for n in nodes])
plot = plot_beliefs(aq, states=model.actions, xlabels_rotation=45,
                    ylabels=[format_history(h) for h in histories], ax=axs[2])
plt.colorbar(plot, ax=axs[2])
axs[2].set_title('Augmented values')


# Plot for observed beliefs
histories, nodes = node_search(pol, order='q')
seen_beliefs = set()
filt_nodes = []
filt_histories = []
for n, h in zip(nodes, histories):
    s = str(n.belief.array)
    if s not in seen_beliefs:
        filt_nodes.append(n)
        filt_histories.append(h)
        seen_beliefs.add(s)
print("Found {} beliefs".format(len(filt_nodes)))
nodes = filt_nodes[:100]
histories = filt_histories[:100]

fig, axs = plt.subplots(1, 2, sharey=True)
b = np.vstack([n.belief.array for n in nodes])
plot = plot_beliefs(b, states=model.states, xlabels_rotation=45, ax=axs[0])
plt.colorbar(plot, ax=axs[0])
axs[0].set_title('Beliefs')

q = np.array([[c.value if c is not None else 0. for c in n.children]
              for n in nodes])
plot = plot_beliefs(q, states=model.actions, xlabels_rotation=45,
                    ylabels=[format_history(h) for h in histories], ax=axs[1])
plt.colorbar(plot, ax=axs[1])
axs[1].set_title('Values')
