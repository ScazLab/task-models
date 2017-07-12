import os
import subprocess
from queue import Empty
from datetime import timedelta
from multiprocessing import Process, cpu_count, Queue

import numpy as np


"""Implements repeating several times the same function through processes
and returning the list of results. Takes care of using a different numpy
random state in each process.
"""


class RepeatPool:

    def __init__(self, target):
        self.n_processes = cpu_count()
        self.target = target

    def work(self, seed):
        np.random.seed(seed)
        result = self.target()
        self.result_queue.put(result)

    def run(self, n):
        self.workers = [None] * self.n_processes
        self.to_go = n  # Number of repetitions still to be run
        self.result_queue = Queue()
        results = []
        while self._still_working():
            self._clean()
            self._fill()
            try:  # Waits for at least one job to push results.
                results.append(self.result_queue.get(False, 1))
            except Empty:  # The timeout prevents blocking if jobs fail to push
                pass
        # Collect remaining results just in case
        results.extend([self.result_queue.get_nowait()
                        for _ in range(n - len(results))])
        return results

    def _still_working(self):
        return self.to_go > 0 or any([w is not None for w in self.workers])

    def _start_or_None(self):
        if self.to_go > 0:
            self.to_go -= 1
            p = Process(target=self.work, args=(np.random.randint(2**32),))
            p.start()
            return p
        else:
            return None

    def _clean(self):
        self.workers = [w if w is None or w.is_alive() else self._clean_job(w)
                        for w in self.workers]

    def _clean_job(self, w):
        if w.exitcode > 0:
            raise RuntimeError("A subprocess has failed.")
        w.join()
        return None

    def _fill(self):
        self.workers = [self._start_or_None()
                        if w is None else w for w in self.workers]


def repeat(func, n):
    p = RepeatPool(func)
    return p.run(n)


def get_process_elapsed_time(pid=None):
    """Return process CPU time from the system."""
    if pid is None:
        pid = os.getpid()
    t = subprocess.check_output(["ps", "-o", "etime=", "-p", str(pid)])
    # Decode from b"[[%d-]%h]:%m:%s"
    t = t.decode().strip().replace('-', ':').split(':')[::-1]  # also reverse
    t += ["0"] * (4 - len(t))  # fill missing optional fields (%d, %h)
    return timedelta(seconds=float(t[0]), minutes=int(t[1]),
                     hours=int(t[2]), days=int(t[3]))
