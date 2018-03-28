==========================================
Task models for human robot collaboration
==========================================
.. image:: https://travis-ci.org/ScazLab/task-models.svg?branch=master
    :target: https://travis-ci.org/ScazLab/task-models

.. image:: https://api.codacy.com/project/badge/Grade/7625ee80663049fd8cb8727c98f6aecc
    :target: https://www.codacy.com/app/Baxter-collaboration/task-models?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ScazLab/task-models&amp;utm_campaign=Badge_Grade

Tools to manipulate and use task models for human robot collaboration.

If you are using this software and or one of its components, we warmly
recommend you to cite the following paper:

.. [Roncone2017] Roncone Alessandro, Mangin Olivier, Scassellati Brian
   **Transparent Role Assignment and Task Allocation in Human Robot
   Collaboration** *IEEE International Conference on Robotics and Automation
   (ICRA 2017)*, Singapore.
   `[PDF] <http://alecive.github.io/papers/[Roncone%20et%20al.%202017]%20Transparent%20Role%20Assignment%20and%20Task%20Allocation%20in%20Human%20Robot%20Collaboration.pdf>`_
   `[BIB] <http://alecive.github.io/papers/[Roncone%20et%20al.%202017]%20Transparent%20Role%20Assignment%20and%20Task%20Allocation%20in%20Human%20Robot%20Collaboration.bib>`_


Repository overview
-------------------

The top-level directories contain the following code:

- :code:`samples`: scripts,
- :code:`tests`: unittests (run :code:`python -m unittest discover tests`),
- :code:`visualization`: task models and policies visualizations based on `d3.js <https://d3js.org/>`.

The code from the :code:`task_models` package contains a set of classes to
represents models of tasks for human robot collaboration and in particular
hierarchical task models. The code mostly consists in the following components:

- The :code:`state.py` and :code:`action.py` modules define useful classes used
  in :code:`task.py`. In addition to providing useful objects to represent
  hierarchies of task (HTM), the latter also implements the techniques for
  extracting such structure that were introduced in [Hayes2016]_.
- The :code:`lib` directory provides:
  - :code:`pomdp.py`: a python wrapper to Anthony Cassandra's POMDP solver. Please visit `pomdp.org <http://www.pomdp.org/>`_.
  - :code:`pomcp.py`: a partial implementation of [Silver2010]_,
  - :code:`belief.py`: belief representations,
  - :code:`py23.py`: compatibility code for python 2 and 3,
  - :code:`utils.py`: additional helpers.
- :code:`utils` is a clone of `<https://github.com/omangin/python-utils>`_.
- :code:`task_to_pomdp.py`: code mostly used for [Roncone2017]_.


.. [Silver2010] Silver, David and Veness, Joel *Monte-Carlo Planning in Large
   POMDPs* (2010) 

.. [Hayes2016] Hayes, Bradley and Scassellati, Brian *Autonomously constructing
   hierarchical task networks for planning and human-robot collaboration*, IEEE
   International Conference on Robotics and Automation (ICRA 2016)

Prerequisites for using the POMDP solvers
-----------------------------------------

This package requires a binary from Anthony Cassandra's POMDP solver. Please visit `pomdp.org <http://www.pomdp.org/>`_ for any matter related to the POMDP solver. In order to be using the *simplex* finite grid method, a fork of the version from `cmansley <https://github.com/cmansley/pomdp-solve>`_ needs to be installed that contains a fix to the original code. You can get the fork `here <https://github.com/scazlab/pomdp-solve>`_.

The python code is looking for the :code:`pomdp-solve` executable in your :code:`$PATH`. Here are some instructions on how to compile and install the solver properly (assuming that :code:`~/src` is the directory in which you usually place your code)::

   cd ~/src
   git clone https://github.com/scazlab/pomdp-solve
   cd pomdp-solve
   mkdir build
   cd build/
   ../configure --prefix=$HOME/.local
   make
   make install

Make sure that :code:`~/.local/bin` is in yout path and now you should have :code:`pomdp-solve` installed in it, and it should be available for the python package to be used.

ICRA 2017
---------

To generate the policy from the experiment in [Roncone2017]_, please use the script :code:`samples/icra_scenario2pomdp.py`. The script will generate the corresponding POMDP model, solve it with Anthony Cassandra's POMDP solver, and store the corresponding policy under :code:`visualization/policy/json/icra.json`. To run the full experiment on the baxter robot, please refer to `github.com/ScazLab/baxter_collaboration <https://github.com/ScazLab/baxter_collaboration>`_.
