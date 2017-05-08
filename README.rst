==========================================
Task models for human robot collaboration
==========================================
.. image:: https://travis-ci.org/ScazLab/task-models.svg?branch=master
    :target: https://travis-ci.org/ScazLab/task-models

Tools to manipulate and use task models for human robot collaboration.

If you are using this software and or one of its components, we warmly recommend you cite the following paper:

    [Roncone2017] Roncone Alessandro, Mangin Olivier, Scassellati Brian **Transparent Role Assignment and Task Allocation in Human Robot Collaboration** *IEEE International Conference on Robotics and Automation (ICRA 2017)*, Singapore.

Prerequisites
-------------

This package requires a binary from Anthony Cassandra's POMDP solver. Please visit `<http://www.pomdp.org/>`_ for any matter related to the POMDP solver.

Here are some instructions on how to compile and install the solver properly (assuming that :code:`~/src` is the directory in which you usually place your code)::

   cd ~/src
   wget www.pomdp.org/code/pomdp-solve-5.4.tar.gz
   tar -xvzf pomdp-solve-5.4.tar.gz
   cd pomdp-solve-5.4/
   mkdir build
   cd build/
   ../configure --prefix=$HOME/.local
   make
   make install


Now you should have :code:`pomdp-solve` installed in your local path, and it should be available for the python package to be used.
