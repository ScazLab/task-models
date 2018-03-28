"""Evaluation of the values of the exploration factor used in POMCP for various
parameters.
"""


import numpy as np
import matplotlib.pyplot as plt


percentages = np.array([1, 5, 10, 30, 50, 70, 100])[:, None]
ns = np.hstack([np.arange(1, 100), np.arange(200, 100000, 100)])

l_ns = np.log(ns)
ys = np.sqrt(l_ns / (percentages * ns))

plt.interactive(True)
plt.semilogx(ns, ys.T)
plt.legend(["{}%".format(p) for p in percentages])
