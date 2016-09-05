import numpy as np

from htm.lib.pomdp import GraphPolicyRunner
from htm.hold_and_release import hold_and_release_pomdp


C_WAIT = 1.
C_FAILURE = 5.
C_COMMUNICATION = 2.


p = hold_and_release_pomdp(C_WAIT, C_FAILURE, C_COMMUNICATION)
pgr = GraphPolicyRunner(p.solve())
print(pgr.gp.to_json(indent=2))
for i in range(10):
    print("|{: 3}. current state: {}, action: {}".format(
        i, pgr.current, pgr.get_action()))
    o = np.random.choice(p.observations)   # TODO actually simulate POMDP
    print("    observed: " + o)
    pgr.step(o)
