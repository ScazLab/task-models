import time
from multiprocessing import Process, cpu_count, Queue


class RepeatPool:

    refresh_period = .01

    def __init__(self, target):
        self.n_processes = cpu_count()
        self.target = target

    def work(self):
        result = self.target()
        self.result_queue.put(result)

    def run(self, n):
        self.workers = [None] * self.n_processes
        self.to_go = n
        self.result_queue = Queue()
        while self._still_working():
            self._clean()
            self._fill()
            time.sleep(self.refresh_period)
        return [self.result_queue.get() for _ in range(n)]

    def _still_working(self):
        return self.to_go > 0 or any([w is not None for w in self.workers])

    def _start_or_None(self):
        if self.to_go > 0:
            self.to_go -= 1
            p = Process(target=self.work)
            p.start()
            return p
        else:
            return None

    def _clean(self):
        self.workers = [w if w is None or w.is_alive() else self._clean_job(w)
                        for w in self.workers]

    def _clean_job(self, w):
        if w.exitcode < 0:
            raise RuntimeError("A subprocess has failed.")
        w.join()
        return None

    def _fill(self):
        self.workers = [self._start_or_None()
                        if w is None else w for w in self.workers]


def repeat(func, n):
    p = RepeatPool(func)
    return p.run(n)
