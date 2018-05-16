import json
from io import open

from task_models.action import Action
from task_models.task import (COMBINATION_CLASSES, AlternativeCombination,
                              HierarchicalTask, LeafCombination,
                              ParallelCombination, SequentialCombination)

test_json = "/home/jake/Code/rpi_integration/tests/out/full_chair.json"

j = json.load(open(test_json, "r", encoding="utf-8"))


def build_htm_recursively(root):
    """Recursively traverses input json file and builds HTM"""
    name = root['name']
    combination = root['combination']

    # Base case, if no children than node is an action
    if not root['children']:
        agent = [a for a in root['attributes'] if not a == 'highlighted']
        return LeafCombination(Action(name=name, agent=agent[0]))

    children = []
    for c in root['children']:
        # Recursive call
        children.append(build_htm_recursively(c))

    # Will wrap subtree in a combination depending on combination attribute
    return COMBINATION_CLASSES[combination](children, name=name)


def json_to_htm(json_path):
    j = json.load(open(json_path, "r", encoding="utf-8"))
    return HierarchicalTask(build_htm_recursively(j['nodes']))
