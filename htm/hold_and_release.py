import numpy as np

from .lib.pomdp import POMDP


def hold_and_release_pomdp(cost_wait, cost_failure, cost_communication,
                           proba_finish=.8):
    start = np.array([1, 0, 0])
    T = np.zeros((3, 3, 3))
    T[:, 0, :] = [proba_finish, 1 - proba_finish, 0.],  # if human has not
    #                                                   # finished
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
    R[0, :, :, :] = cost_wait  # Cost for waiting is constant
    R[1, 0, :, :] = [[cost_failure], [0], [0]]  # On physical action, failure
    #                                           # if human unfinished
    R[2, :, :, :] = cost_communication  # Constant cost for communication
    return POMDP(T, O, R, start, 1,
                 actions=['wait', 'physical', 'communicate'],
                 states=['zero', 'one', 'final'],
                 observations=['nothing', 'doing', 'done'],
                 values='cost')
