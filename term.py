"""Set of commands for pretty prints.
"""


from __future__ import print_function

import sys
try:
    from termcolor import colored
except ImportError:
    def colored(s, *args, **kwargs):
        return s


def clear_progress_bar(done, todo, count=True, size=50, color=None, prefix=''):
    length = 7 + size + 12 * count + len(prefix)
    print(length * ' ', end='\r')
    sys.stdout.flush()


def print_progress_bar(done, todo, count=True, size=50, color=None, prefix=''):
    if todo < done:
        return
    s = progress_bar(done, todo, count=count, size=size)
    print(prefix + colored(s, color=color), end='\r')
    if todo == done:
        print()
    sys.stdout.flush()


def progress_bar(done, todo, count=True, size=50):
    count_str = ' {:{digits},d}/{:{digits},d}'.format(
        done, todo, digits=int(todo / 10)
        ) if count else ''
    return '[{done_bar:.<{size}} {percent:>3d}%] {count}'.format(
        done_bar='#' * int(size * done / todo),
        size=size,
        percent=int(100 * done / todo),
        count=count_str)
