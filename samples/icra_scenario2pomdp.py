import os

from htm.task_to_pomdp import chair_task, T_WAIT, T_COMM, C_INTR
from htm.stool_scenarios import HTMToPOMDP


## Tested scenarios:
# 1. with full sequence of sequential actions
R_END = 0.1
LOOP = False
# 2. with full sequence of sequential actions
# R_END = 100
# LOOP = True

## Convert the task into a POMDP

h2p = HTMToPOMDP(T_WAIT, T_COMM, C_INTR, end_reward=R_END, loop=LOOP)
p = h2p.task_to_pomdp(chair_task)
#p.discount = .99

gp = p.solve(method='grid', n_iterations=500, verbose=True)
gp.dump_to(os.path.join(os.path.dirname(__file__),
                        '../visualization/policy/json/test.json'))

from htm.lib.pomdp import GraphPolicyBeliefRunner

pol = GraphPolicyBeliefRunner(gp, p)
pol.save_trajectories_from_starts(
    os.path.join(os.path.dirname(__file__),
                 '../visualization/trajectories/json/trajectories.json'),
    horizon=10, indent=2)
gp2 = pol.visit()
gp2.dump_to(os.path.join(os.path.dirname(__file__),
                         '../visualization/policy/json/from_beliefs.json'))


from htm.plot import plot_beliefs
import matplotlib.pyplot as plt

plt.interactive(True)
plt.close('all')


def plot_values(values, actions, p):
    b = values - values.min()
    b /= b.max(-1)[:, None]
    plot = plot_beliefs(b, states=p.states, xlabels_rotation=45,
                        ylabels=["{}: {}".format(i, a)
                                for i, a in enumerate(actions)])
    plt.colorbar(plot)


plt.figure()
plot_values(gp.values, gp.actions, p)
plt.title('Policy values')

plt.figure()
plot_values(gp2.values, gp2.actions, p)
plt.title('Belief trajectory')
