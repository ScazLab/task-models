import os

from task_models.task_to_pomdp import HTMToPOMDP
from task_models.stool_scenarios import (stool_task_sequential, T_WAIT, T_ASK,
                                         T_TELL, C_INTR)


## Tested scenarios:
# 1. with full sequence of sequential actions
R_END = 0.1
LOOP = False
# 2. with full sequence of sequential actions
# R_END = 100
# LOOP = True
PLOT = False

## Convert the task into a POMDP

h2p = HTMToPOMDP(T_WAIT, T_ASK, T_TELL, C_INTR, end_reward=R_END, loop=LOOP)
p = h2p.task_to_pomdp(stool_task_sequential)
#p.discount = .99

gp = p.solve(method='grid', n_iterations=500, verbose=True)
gp.save_as_json(os.path.join(os.path.dirname(__file__),
                             '../visualization/policy/json/icra.json'))

from task_models.lib.pomdp import GraphPolicyBeliefRunner

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


def plot_values(values, actions, p):
    b = values - values.min()
    b /= b.max(-1)[:, None]
    plot = plot_beliefs(b, states=p.states, xlabels_rotation=45,
                        ylabels=["{}: {}".format(i, a)
                                for i, a in enumerate(actions)])
    plt.colorbar(plot)


if PLOT:
    from task_models.plot import plot_beliefs
    import matplotlib.pyplot as plt

    plt.interactive(True)
    plt.close('all')




    plt.figure()
    plot_values(gp.values, gp.actions, p)
    plt.title('Policy values')

    plt.figure()
    plot_values(gp2.values, gp2.actions, p)
    plt.title('Belief trajectory')
