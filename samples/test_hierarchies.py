from task_models.state import State
from task_models.task import AbstractAction, TaskGraph, MetaAction


b1 = AbstractAction('b1')
b2 = AbstractAction('b2')
b3 = AbstractAction('b3')
c1 = AbstractAction('c1')
c2 = AbstractAction('c2')

actions = {a.name: a for a in [b1, b2, b3, c1, c2]}


class HState(State):

    def __init__(self, history):
        self.h = ':'.join(history)

    def __hash__(self):
        return self.h.__hash__()

    def __eq__(self, other):
        return isinstance(other, HState) and self.h == other.h

    def __repr__(self):
        return 'HState<{}>'.format(self.h)


def history_states_path(history):
    history = history.split(':')
    path = []
    for i, h in enumerate(history):
        path.append(HState(history[:i]))
        path.append(actions[h])
    path.append(HState(history))
    return path


g = TaskGraph()
g.add_path(history_states_path('b1:b2:b3:c1:c2'))
g.add_path(history_states_path('c1:c2:b1:b2:b3'))
cg = g.conjugate()

while len(cg.all_nodes()) > 1:
    print('Conjugate graph:')
    for (s, l, d) in cg.all_transitions():
        print("  {} -> {}".format(s, d))
    print()
    try:
        chain = next(cg.get_max_chains())
        cg.compact(chain, MetaAction('sequence', chain))
    except StopIteration:
        try:
            clique = next(cg.get_max_cliques())
            cg.compact(clique, MetaAction('parallel', clique))
        except StopIteration:
            break


print('Compacted conjugate graph')
for (s, l, d) in cg.all_transitions():
    print("  {} -> {}".format(s, d))
