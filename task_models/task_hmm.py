from __future__ import unicode_literals
import numpy as np
import itertools as iter
import os
import errno
import re
from abc import ABCMeta, abstractmethod
from .task import HierarchicalTask, \
    LeafCombination, SequentialCombination, \
    ParallelCombination, AlternativeCombination, \
    AbstractAction


def unique_rows(a):
    b = a.ravel().view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, unique_idx = np.unique(b, return_index=True)
    return a[unique_idx]


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


class HierarchicalTaskHMM(HierarchicalTask):
    """Tree representing a hierarchy of tasks which leaves are actions."""

    _metaclass_ = ABCMeta

    def __init__(self, root=None, name=None, num_feats_action=None,
                 feats='cum', supp_bhvs=None):
        super(HierarchicalTaskHMM, self).__init__(root)
        self.name = name
        self.num_obj = 0
        self.obj_names = dict()
        self.feat_names = dict()
        self.feats = feats
        self.num_feats_action = num_feats_action
        self.supp_bhvs = supp_bhvs
        self.all_trajectories = []
        self.bin_trajectories = []
        self.bin_trajectories_test = []
        self.train_set_actions = []
        self.train_set_traj = []
        self.test_set_actions = []
        self.test_set_traj = []
        self.train_set_sb = []
        self.train_set_sb_actions = []
        self.train_set_sb_traj = []
        self.test_set_sb = []
        self.test_set_sb_traj = []
        self.test_set_sb_actions = []
        self.gen_dict()

    def gen_dict(self):
        regexp = r'( order)'
        self._gen_dict_rec(self.root, regexp)
        if self.feats == 'cum' or self.feats == 'shared':
            for feat_idx in range(self.num_feats_action):
                self.feat_names["feat_" + str(feat_idx)] = feat_idx
        else:
            self.feat_names = self.obj_names

    def _gen_dict_rec(self, node, regexp):
        if isinstance(node, LeafCombination):
            name = node.name
            rem = re.search(regexp, node.name)
            if rem:
                name = node.name[:rem.start()]
            if not(name in self.obj_names.keys()):
                self.obj_names[name] = self.num_obj
                self.num_obj += 1
        else:
            for child in node.children:
                self._gen_dict_rec(child, regexp)

    def _get_cum_feats_keys(self, traj):
        regexp = r'( order)'
        keys = [self.obj_names.get(key.name) if re.search(regexp, key.name) is None
                else self.obj_names.get(key.name[:re.search(regexp, key.name).start()])
                for key in traj]
        return keys

    @staticmethod
    def base_name(name):
        regexp = r'( order)'
        rem = re.search(regexp, name)
        if rem:
            name = name[:rem.start()]
        return name

    def gen_all_trajectories(self):
        self.all_trajectories = \
            self._gen_trajectories_rec(self.root)

    def _gen_trajectories_rec(self, node):
        """Generates all possible trajectories from an HTM.

        :returns: list of tuples of the following form
        [(proba, [LC1, LC2, ...]), (), ...]
        Each tuple represents a trajectory of the HTM,
        with the first element equal to the probability of that trajectory
        and the second element equal to the list of leaf combinations.
        """
        if isinstance(node, LeafCombination):
            return [(1, [node])]
        elif isinstance(node, ParallelCombination):
            return self._gen_trajectories_rec(node.to_alternative())
        elif isinstance(node, AlternativeCombination):
            children_trajectories = [(p * node.proba[c_idx], seq)
                                     for c_idx, c in enumerate(node.children)
                                     for p, seq in self._gen_trajectories_rec(c)]
            return children_trajectories
        elif isinstance(node, SequentialCombination):
            children_trajectories = [
                self._gen_trajectories_rec(c)
                for c_idx, c in enumerate(node.children)]
            new_trajectories = []
            product_trajectories = list(
                iter.product(*children_trajectories))
            for product in product_trajectories:
                probas, seqs = zip(*product)
                new_trajectories.append((float(np.product(probas)),
                                         list(iter.chain.from_iterable(seqs))))
            return new_trajectories
        else:
            raise ValueError("Reached invalid type during recursion.")

    def gen_bin_feats_test_set(self, test_set_size):
        for i in range(test_set_size):
            self.bin_trajectories_test.append(self.bin_trajectories[i])
        self.bin_trajectories_test = np.array(self.bin_trajectories_test)

    def gen_bin_feats_traj(self):
        """Generates binary features for all possible trajectories from an HTM
        and writes them to file.
        Should only be called after calling gen_all_trajectories().
        """
        if self.all_trajectories:
            regexp = r'( order)'
            trajectories = list(zip(*self.all_trajectories))[1]
            bin_feats_init = np.array([0] * len(self.obj_names))
            for traj_idx, traj in enumerate(trajectories):
                bin_feats = np.tile(bin_feats_init,
                                    (len(set(traj)) + 1, 1))
                for node_idx, node in enumerate(traj):
                    if self.feats == 'cum':
                        # keys = [self.obj_names.get(key.name) if re.search(regexp, key.name) is None
                        #         else self.obj_names.get(key.name[:re.search(regexp, key.name).start()])
                        #         for key in traj[:node_idx + 1]]
                        keys = self._get_cum_feats_keys(traj[:node_idx + 1])
                        bin_feats[node_idx + 1, keys] = 1
                    else:
                        action = self.obj_names.get(traj[node_idx].name)
                        if action is None:
                            action = self.obj_names.get(traj[node_idx]
                                                        .name[:re.search(regexp, traj[node_idx].name).start()])
                        bin_feats[node_idx + 1, action] = 1
                #bin_feats = unique_rows(bin_feats)
                self.bin_trajectories.append(bin_feats)
            self.bin_trajectories = np.array(self.bin_trajectories)
        else:
            raise ValueError("Cannot generate bin feats before generating all trajectories.")

    @staticmethod
    def gen_object_bin_feats_sb(num_feats, feat_probs):
        """Generates binary features only for an object from the HTM.

        :param num_feats: number of features this object has (for now, same for all objects)
        :param feat_probs: list of probabilities of length = num_feats, where each prob
        is used to generate each of the binary features for this object (for now, each prob
        is generated using a uniform distribution)

        :returns: list of num_feats binary features generated based on the feat_probs list
        """
        if num_feats != len(feat_probs):
            raise ValueError("num_feats != len(feat_probs)"
                             "You should pass in prob values for all features.")
        bin_feats_obj = np.random.binomial(1, feat_probs, size=num_feats)
        return bin_feats_obj

    def gen_training_set_actions(self, size, bias=False):
        if len(self.all_trajectories) == 0:
            self.gen_all_trajectories()
        self.train_set_actions, self.train_set_traj = \
            self._gen_set_bin_feats_actions(self.all_trajectories, size, bias=bias)

    def gen_test_set_actions(self, size):
        if len(self.train_set_actions) == 0:
            raise ValueError("There exists no training set. Cannot create test set.")
        self.test_set_actions, self.test_set_traj = \
            self._gen_set_bin_feats_actions(self.all_trajectories, size)

    def _gen_set_bin_feats_actions(self, trajectories, size,
                                   bias=False, bias_weight=50):
        """Generates sets to be used for a dataset only for actions.
        It excludes supportive behaviors."""
        len_traj = len(trajectories)
        set_bin_feats = []
        regexp = r'( order)'

        if self.feats == 'cum' or self.feats == 'shared':
            traj_bin_feats_init = np.array([0] * self.num_feats_action)
        else:
            traj_bin_feats_init = np.array([0] * (self.num_obj*self.num_feats_action))
        set_traj_local = []
        bias_progress = 0
        for i in range(size):
            traj = trajectories[np.random.randint(0, len_traj)][1]
            if bias is True and bias_progress <= bias_weight:
                zero_leaf_name = self.base_name(traj[0].name)
                fourth_leaf_name = self.base_name(traj[4].name)
                while zero_leaf_name != 'gatherparts_leg_3' \
                        or fourth_leaf_name != 'gatherparts_leg_1':
                    traj = trajectories[np.random.randint(0, len_traj)][1]
                    zero_leaf_name = self.base_name(traj[0].name)
                    fourth_leaf_name = self.base_name(traj[4].name)
                bias_progress += 1
            set_traj_local.append(traj)
            traj_bin_feats = np.tile(traj_bin_feats_init,
                                     (len(set(traj)) + 1, 1))
            for node_idx, node in enumerate(traj):
                if self.feats == 'cum':
                    keys = self._get_cum_feats_keys(traj[:node_idx + 1])
                    traj_bin_feats[node_idx + 1, keys] = 1
                else:
                    node_bin_feats_sb = self.gen_object_bin_feats_sb(
                        node.action.num_feats, node.action.feat_probs)
                    if self.feats == 'shared':
                        traj_bin_feats[node_idx + 1, :] = node_bin_feats_sb
                    else:
                        action = self.obj_names.get(traj[node_idx].name)
                        if action is None:
                            action = self.obj_names.get(traj[node_idx].name[:re.
                                                        search(regexp, traj[node_idx].name).start()])
                        start_idx = action * self.num_feats_action
                        end_idx = start_idx + self.num_feats_action
                        traj_bin_feats[node_idx + 1, start_idx:end_idx] = node_bin_feats_sb
                # traj_bin_feats = \
                #     [self.gen_object_bin_feats_sb(leaf.action.num_feats, leaf.action.feat_probs)
                #      for leaf in traj]
                #traj_bin_feats.insert(0, np.array([0] * self.num_feats_action))
            set_bin_feats.append(traj_bin_feats)
        return np.array(set_bin_feats), set_traj_local

    def gen_train_set_sb(self, user_prefs, num_dems):
        self.train_set_sb, self.train_set_sb_traj, self.train_set_sb_actions = \
            self._gen_set_sb(self.train_set_traj, self.train_set_actions,
                             user_prefs, num_dems)

    def gen_test_set_sb(self, user_prefs, num_dems):
        self.test_set_sb, self.test_set_sb_traj, self.test_set_sb_actions = \
            self._gen_set_sb(self.test_set_traj, self.test_set_actions, user_prefs,
                             num_dems)

    def _gen_set_sb(self, trajectories, traj_actions, users_prefs, num_dems):
        len_train_traj = len(trajectories)
        if len_train_traj == 0:
            raise ValueError("Please generate training set w/ actions "
                             "before generating training supportive behavior set.")
        set_sb = []
        set_sb_traj = []
        set_sb_actions = []
        for user in sorted(users_prefs.keys()):
            all_dems_user_bin_feats, user_sb_traj, user_sb_actions = \
                self._gen_user_sb(trajectories, traj_actions, users_prefs[user], num_dems)
            set_sb.append(all_dems_user_bin_feats)
            set_sb_traj.append(user_sb_traj)
            set_sb_actions.append(user_sb_actions)
        return np.array(set_sb), np.array(set_sb_traj), np.array(set_sb_actions)

    @abstractmethod
    def _gen_user_sb(self, trajectories, traj_actions, users_prefs, num_dems):
        pass

    def reset_sb_sets(self):
        self.train_set_sb = []
        self.train_set_sb_actions = []
        self.train_set_sb_traj = []
        self.test_set_sb = []
        self.test_set_sb_traj = []
        self.test_set_sb_actions = []

    def reset_main_sets(self):
        self.all_trajectories = []
        self.bin_trajectories = []
        self.bin_trajectories_test = []
        self.train_set_actions = []
        self.train_set_traj = []
        self.test_set_actions = []
        self.test_set_traj = []


class HierarchicalTaskHMMSuppSimple(HierarchicalTaskHMM):
    """Extension to the HMM task to the rd naming
    convention for supportive behaviors.
    """

    def __init__(self, *args, **kwargs):
        super(HierarchicalTaskHMMSuppSimple, self).__init__(*args, **kwargs)

    def _gen_user_sb(self, trajectories, traj_actions, user_prefs, num_dems):
        len_train_traj = len(trajectories)
        all_dems_user_bin_feats = []
        regexp = r'( order)'
        user_sb_traj = []
        user_sb_actions = []
        for i in range(num_dems):
            traj_idx = np.random.randint(0, len_train_traj)
            traj = trajectories[traj_idx]
            user_sb_traj.append(trajectories[traj_idx])
            user_sb_actions.append(traj_actions[traj_idx])
            traj_user_bin_feats = []
            for leaf in traj:
                name = leaf.name
                rem = re.search(regexp, leaf.name)
                if rem:
                    name = leaf.name[:rem.start()]
                traj_user_bin_feats.append(user_prefs[name])
            traj_user_bin_feats.insert(0, np.array([0] * len(self.supp_bhvs)))
            all_dems_user_bin_feats.append(traj_user_bin_feats)
        return all_dems_user_bin_feats, user_sb_traj, user_sb_actions


class HierarchicalTaskHMMSuppNamed(HierarchicalTaskHMM):
    """Extension to the HMM task to the rd naming
    convention for supportive behaviors.
    """

    def __init__(self, *args, **kwargs):
        super(HierarchicalTaskHMMSuppNamed, self).__init__(*args, **kwargs)

    def _gen_user_sb(self, trajectories, traj_actions, user_prefs, num_dems):
        len_train_traj = len(trajectories)
        len_traj = traj_actions.shape[1]
        all_dems_user_bin_feats = []
        regexp = r'( order)'
        user_sb_traj = []
        user_sb_actions = []
        for i in range(num_dems):
            traj_idx = np.random.randint(0, len_train_traj)
            traj = trajectories[traj_idx]
            user_sb_traj.append(trajectories[traj_idx])
            user_sb_actions.append(traj_actions[traj_idx])
            traj_user_bin_feats = np.zeros((len_traj, len(self.supp_bhvs)))
            num_leg_leaves = 0
            for leaf_idx, leaf in enumerate(traj):
                name = leaf.name
                rem = re.search(regexp, leaf.name)
                if rem:
                    name = leaf.name[:rem.start()]
                if name.startswith('bring_leg'):
                    num_leg_leaves += 1
                for key in user_prefs.keys():
                    if (name in user_prefs[key]) \
                        or ('end' in user_prefs[key] and leaf_idx == len_traj-2)  \
                        or ('all_bring_leg' in user_prefs[key] and name.startswith('bring_leg')) \
                        or (sum(['time_leg' in leg for leg in user_prefs[key]]) >= 1
                            and int(user_prefs[key][0].split('_')[2]) == num_leg_leaves):
                        traj_user_bin_feats[leaf_idx+1, self.supp_bhvs[key]] = 1
            all_dems_user_bin_feats.append(traj_user_bin_feats)
        return all_dems_user_bin_feats, user_sb_traj, user_sb_actions


class HierarchicalTaskHMMSuppRD(HierarchicalTaskHMM):
    """Extension to the HMM task to the rd naming
    convention for supportive behaviors.
    """

    def __init__(self, *args, **kwargs):
        super(HierarchicalTaskHMMSuppRD, self).__init__(*args, **kwargs)

    def _gen_user_sb(self, trajectories, traj_actions, user_prefs, num_dems):
        len_train_traj = len(trajectories)
        len_traj = traj_actions.shape[1]
        all_dems_user_bin_feats = []
        regexp = r'( order)'
        user_sb_traj = []
        user_sb_actions = []
        for i in range(num_dems):
            traj_idx = np.random.randint(0, len_train_traj)
            traj = trajectories[traj_idx]
            user_sb_traj.append(trajectories[traj_idx])
            user_sb_actions.append(traj_actions[traj_idx])
            traj_user_bin_feats = np.zeros((len_traj, len(self.supp_bhvs)))
            num_gp_leg_leaves = 0
            num_gp_front_leg_leaves = 0
            num_gp_back_leg_leaves = 0
            num_ass_leg_leaves = 0
            num_ass_front_leg_leaves = 0
            num_ass_back_leg_leaves = 0
            first_time = True
            for leaf_idx, leaf in enumerate(traj):
                name = leaf.name
                rem = re.search(regexp, leaf.name)
                if rem:
                    name = leaf.name[:rem.start()]
                name_comp = name.split('_')
                if name.startswith('gatherparts_leg'):
                    num_gp_leg_leaves += 1
                    if name_comp[2] in ['1', '2']:
                        num_gp_front_leg_leaves += 1
                    elif name_comp[2] in ['3', '4']:
                        num_gp_back_leg_leaves += 1
                elif name.startswith('assemble_leg'):
                    num_ass_leg_leaves += 1
                    if name_comp[2] in ['1', '2']:
                        num_ass_front_leg_leaves += 1
                    elif name_comp[2] in ['3', '4']:
                        num_ass_back_leg_leaves += 1
                for key in user_prefs.keys():
                    change = False
                    if sum(['time_assemble_leg' in leg for leg in user_prefs[key]]) >= 1 \
                            and int(user_prefs[key][0].split('_')[3]) == num_ass_leg_leaves \
                            and first_time is True:
                        change = True
                        first_time = False
                    if (name in user_prefs[key]) \
                        or ('end' in user_prefs[key] and leaf_idx == len_traj-2)  \
                        or ('all_gatherparts_leg' in user_prefs[key] and name.startswith('gatherparts_leg')) \
                        or (change is True) \
                        or (sum(['all_gatherparts_leg' in leg for leg in user_prefs[key]]) >= 1
                            and name.startswith('gatherpart_leg')) \
                        or (sum(['all_assemble_leg' in leg for leg in user_prefs[key]]) >= 1
                            and name.startswith('assemble_leg')):
                        traj_user_bin_feats[leaf_idx+1, self.supp_bhvs[key]] = 1
            all_dems_user_bin_feats.append(traj_user_bin_feats)
        return all_dems_user_bin_feats, user_sb_traj, user_sb_actions


class PredAction(AbstractAction):

    """Action abstraction to be used for the prediction project.

    :param name: str (must be unique)
    :param durations: (float: human, float: robot, float: error)
        Duration for human and robot to perform the action as well as error
        time when robot starts action at wrong moment (the error time includes
        the time of acting before interruption as well as the recovery time).
    :param human_probability: float
        Probability that the human would take care of this action. If not
        defined, will have to be estimated.
    :param fail_probability: float
        Probability that the robot action fails.
    :param no_probability: float
        Probability that the human answers no to the robot asking if he can
        take the action.
    """

    def __init__(self, name, num_feats, feat_probs):
        super(PredAction, self).__init__(name=name)
        self.num_feats = num_feats
        self.feat_probs = feat_probs

        @property
        def feat_probs(self):
            return self._feat_probs

        @feat_probs.setter
        def feat_probs(self, feat_probs):
            if self.num_feats != len(feat_probs):
                raise ValueError("num_feats != len(feat_probs). "
                                 "Should have prob values for each feature.")
            else:
                self._feat_probs = feat_probs

    def copy(self, rename_format='{}'):
        return PredAction(rename_format.
                          format(self.name), self.num_feats, self.feat_probs)
