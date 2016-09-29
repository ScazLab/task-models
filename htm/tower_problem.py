from htm.task import (HierarchicalTask, LeafCombination, SequentialCombination,
                      AlternativeCombination)
from htm.bring_next_to_pomdp import CollaborativeAction


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
