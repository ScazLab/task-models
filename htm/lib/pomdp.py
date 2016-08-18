import numpy as np


class ValueFunctionParseError(ValueError):
    pass


def parse_value_function(reader):
    has_action = False
    actions = []
    vectors = []
    for line in reader:
        if not line.isspace():
            if has_action:
                # expect vector
                vectors.append(np.fromstring(line, sep=' '))
                has_action = False
            else:
                # expect action
                actions.append(int(line))
                has_action = True
        # else: skip line
    if has_action:
        raise ValueFunctionParseError('Action defined but no vectors follows.')
    return actions, vectors


def parse_policy_graph(reader):
    actions = []
    transitions = []
    for i, line in enumerate(reader):
        line = line.rstrip('\n').rstrip()
        if not line.isspace():
            # 'N A  Z1 Z2 Z3'.split(' ') -> ['N', 'A', '', 'Z1', 'Z2', 'Z3']
            l = line.split(' ')
            n = int(l[0])  # Node name
            assert(n == i)
            actions.append(int(l[1]))
            transitions.append([None if x == '-' else int(x) for x in l[3:]])
    return actions, transitions
