from task_models.utils.multiprocess import repeat, get_process_elapsed_time
from task_models.lib.pomcp import NTransitionsHorizon


class FinishedOrNTransitionsHorizon(NTransitionsHorizon):

    def __init__(self, model, n):
        super(FinishedOrNTransitionsHorizon, self).__init__(n)
        self.model = model
        self._is_final = False

    def is_reached(self):
        return self.n <= 0 or self._is_final

    def decrement(self, a, s, new_s, o):
        super(FinishedOrNTransitionsHorizon, self).decrement(a, s, new_s, o)
        self._is_final = self.model._int_to_state(new_s).is_final()

    def copy(self):
        return FinishedOrNTransitionsHorizon(self.model, self.n)

    @classmethod
    def generator(cls, model, n=100):
        return cls._Generator(cls, model, n)


def episode_summary(model, full_return, h_s, h_a, h_o, h_r, n_calls,
                    elapsed=None):
    indent = 4 * " "
    return ("Evaluation: {} transitions, return: {:4.0f} [{:,} calls in {}]\n"
            "".format(len(h_a), full_return, n_calls, elapsed) +
            "".join(["{ind}{}: {} → {} [{}]\n".format(model._int_to_state(s),
                                                      model.actions[a],
                                                      model.observations[o],
                                                      r, ind=indent)
                     for s, a, o, r in zip(h_s, h_a, h_o, h_r)]) +
            "{ind}{}".format(model._int_to_state(h_s[-1]), ind=indent))


def simulate_one_evaluation(model, pol, max_horizon=200, logger=None):
    init_calls = model.n_simulator_calls
    pol.reset()
    # History init
    h_s = [model.sample_start()]
    h_a = []
    h_o = []
    h_r = []
    horizon = FinishedOrNTransitionsHorizon(model, max_horizon)
    full_return = 0
    while not horizon.is_reached():
        a = model.actions.index(pol.get_action())
        h_a.append(a)
        s, o, r = model.sample_transition(a, h_s[-1])  # real transition
        h_o.append(o)
        h_r.append(r)
        horizon.decrement(a, h_s[-1], s, o)
        pol.step(model.observations[o])
        h_s.append(s)
        full_return = r + model.discount * full_return
    elapsed = get_process_elapsed_time()
    n_calls = model.n_simulator_calls - init_calls
    if logger is not None:
        logger(episode_summary(model, full_return, h_s, h_a, h_o, h_r, n_calls,
                               elapsed=elapsed))
    return {'return': full_return,
            'states': h_s,
            'actions': h_a,
            'observations': h_o,
            'rewards': h_r,
            'elapsed-time': elapsed.total_seconds(),
            'simulator-calls': n_calls,
            }


def evaluate(model, pol, n_evaluation, logger=None):
    def func():
        return simulate_one_evaluation(model, pol, logger=logger)

    return repeat(func, n_evaluation)
