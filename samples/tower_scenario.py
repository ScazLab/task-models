import os

from htm.lib.pomdp import GraphPolicyBeliefRunner
from htm.task import (HierarchicalTask, LeafCombination, SequentialCombination,
                      AlternativeCombination)
from htm.bring_next_to_pomdp import CollaborativeAction, HTMToPOMDP


class TowerProblem:

    def __init__(self, height=2, n_towers=3):
        self.h = height
        self.n = n_towers

    @property
    def n_parts(self):
        return self.h * self.n

    @property
    def parts(self):
        return [str(i) for i in range(self.n_parts)]

    def v_first(self):
        return range(self.n_parts)

    def h_first(self):
        return [i * self.h + h for h in range(self.h) for i in range(self.n)]

    def sequential_combination_from_order(self, order, name=None):
        children = [LeafCombination(CollaborativeAction(
            str(i) + ('-' + name if name is not None else ''),
            str(i))) for i in order]
        return SequentialCombination(children, name=name)

    def alternative_combination_from_orders(self, orders, names=None):
        if names is None:
            names = [str(i) for i, _ in enumerate(orders)]
        return AlternativeCombination([
            self.sequential_combination_from_order(o, n)
            for o, n in zip(orders, names)], name='Alt')

    def task_from_orders(self, orders, names=None):
        return HierarchicalTask(
            root=self.alternative_combination_from_orders(orders, names=names)
            )

    def vertical_horizontal_task(self):
        return self.task_from_orders([self.v_first(), self.h_first()],
                                     names=['vertical', 'horizontal'])


# 1  3  5
# 0  2  4

tp = TowerProblem()
task_vh = tp.vertical_horizontal_task()
orders_medium = [
    [2, 0, 4, 3, 1, 5],
    [2, 4, 0, 3, 5, 1],
    tp.v_first(),
    [0, 1, 4, 5, 2, 3],
    [2, 3, 0, 1, 4, 5],
    [2, 3, 4, 5, 0, 1],
    ]
task_medium = tp.task_from_orders(
        orders_medium,
task = tp.vertical_horizontal_task()
h2p = HTMToPOMDP(2., 8., 5., tp.parts, end_reward=50., discount=.9)

task = task_medium
for o in orders_medium:
    print(o)

p = h2p.task_to_pomdp(task)
gp = p.solve(method='grid', grid_type='pairwise', n_iterations=50,
             verbose=True)
gp.save_as_json(os.path.join(os.path.dirname(__file__),
                             '../visualization/policy/json/test.json'))
pol = GraphPolicyBeliefRunner(gp, p)
pol.save_trajectories_from_starts(
    os.path.join(
        os.path.dirname(__file__),
        '../visualization/trajectories/json/trajectories.json'),
    horizon=10, indent=2)
gp2 = pol.visit()
gp2.save_as_json(os.path.join(
    os.path.dirname(__file__),
    '../visualization/policy/json/from_beliefs.json'))
