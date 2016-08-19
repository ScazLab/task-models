import numpy as np

from htm.lib.pomdp import POMDP


C_WAIT = 1.
C_FAILURE = 5.
C_COMMUNICATION = 2.

start = np.array([1, 0, 0])
T = np.zeros((3, 3, 3))
T[:, 0, :] = [.8, .2, 0.],   # if human has not finished
T[:, 2, :] = [0., 0., 1.],   # if human and robot have finished
T[:, 1, :] = [[0., 1., 0.],  # if human has finished robot needs to act
              [0., 0., 1.],
              [0., 1., 0.]]
O = np.zeros((3, 3, 3))
O[0, :, :] = [1., 0., 0.]         # nothing observed on wait
O[[1, 2], :, :] = [[0., 1., 0.],  # on com or physical human progress is
                   [0., 0., 1.],  # observed
                   [1., 0., 0.]]
R = np.zeros((3, 3, 3, 3))
R[0, :, :, :] = C_WAIT  # Cost for waiting is constant
R[1, 0, :, :] = [[C_FAILURE], [0], [0]]  # On phy., failure if human unfinished
R[2, :, :, :] = C_COMMUNICATION  # Constant cost for communication


p = POMDP(T, O, R, start, 1,
          actions=['wait', 'physical', 'communicate'],
          states=['zero', 'one', 'final'],
          observations=['nothing', 'doing', 'done'],
          values='cost')
actions, vf, pg = p.solve()
print('Actions:', actions)
print('Value functions:')
print(np.vstack(vf))
print('Policy graph:')
print(np.asarray(pg))
