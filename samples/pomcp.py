from __future__ import unicode_literals
import os
import json
from heapq import heappush, heappop

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
        yield next(it), next(it)


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
R_END = 1000.
LOOP = False
R_STATE = True
# 2. with full sequence of sequential actions
# R_END = 100
# LOOP = True
R_SUBTASK = 100

HORIZON = 100
ITERATIONS = 100
EXPLORATION = 1  # 1000
RELATIVE_EXPLO = True  # In this case use smaller exploration
BELIEF_VALUES = True

## Convert the task into a POMDP

h2p = HTMToPOMDP(T_WAIT, T_ASK, T_TELL, intr_cost=C_INTR, end_reward=R_END,
                 loop=LOOP, reward_state=R_STATE, subtask_reward=R_SUBTASK)
p = h2p.task_to_pomdp(stool_task_sequential)
#p.discount = .99

pol = POMCPPolicyRunner(p, iterations=ITERATIONS, horizon=HORIZON,
                        exploration=EXPLORATION,
                        relative_exploration=RELATIVE_EXPLO,
                        belief_values=BELIEF_VALUES)
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
