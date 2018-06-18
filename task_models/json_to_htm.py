import json
from io import open

from task_models.action import Action
from task_models.task import (COMBINATION_CLASSES, AlternativeCombination,
                              HierarchicalTask, LeafCombination,
                              ParallelCombination, SequentialCombination)


def build_htm(node):
    """Recursively traverses input json file and builds HTM"""
    name = node['name']
    combination = node['combination']
    idx = node['id']
    parent = node['parent']

    # Base case, if no children than node is an action
    if not node['children']:
        agent = [a for a in node['attributes'] if not a == 'highlighted']
        return LeafCombination(Action(name=name, agent=agent[0]), idx=idx, parent=parent)

    children = []
    for c in node['children']:
        # Recursive call
        children.append(build_htm(c))

    # Will wrap subtree in a combination depending on combination attribute
    return COMBINATION_CLASSES[combination](children, name=name, idx=idx, parent=parent)


def json_to_htm(json_path):
    j = json.load(open(json_path, "r", encoding="utf-8"))
    return HierarchicalTask(build_htm(j['nodes']))
