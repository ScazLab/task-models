import numpy as np
import matplotlib.pyplot as plt


def plot_beliefs(array, states=None, ylabels=None, xlabels_rotation=0,
                 **kwargs):
    ax = kwargs.pop('ax', None) or plt.gca()
    n_beliefs, n_states = array.shape
    if states is None:
        states = [str(i) for i in range(n_states)]
    assert(len(states) == n_states)
    if ylabels is None:
        ylabels = [str(i) for i in range(n_beliefs)]
    assert(len(ylabels) == n_beliefs)
    if 'cmap' not in kwargs:
        kwargs['cmap'] = plt.cm.Blues
    # Plot
    p = ax.pcolormesh(array, **kwargs)
    # Set ticks
    ax.set_xticks(np.arange(0.5, n_states + 0.5))
    ax.set_xticklabels(states, rotation=xlabels_rotation, ha='right')
    ax.set_yticks(np.arange(0.5, n_beliefs + 0.5))
    ax.set_yticklabels(ylabels)
    ax.set_xlim([0, n_states])
    ax.set_ylim([0, n_beliefs])
    ax.grid()
    return p
