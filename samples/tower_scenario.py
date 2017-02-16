import os

from task_models.lib.pomdp import GraphPolicyBeliefRunner
from task_models.bring_next_to_pomdp import task_modelsToPOMDP
from task_models.tower_problem import TowerProblem


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
    names=['-'.join([str(i) for i in o]) for o in orders_medium])
h2p = HTMToPOMDP(2., 8., 5., tp.parts, end_reward=50., loop=False)

task = task_medium
for o in orders_medium:
    print(list(o))

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
