from __future__ import unicode_literals
import numpy as np
import itertools as iter
import os
import errno
import logging
import re
import shutil
import json

from itertools import permutations

from task_models.state import State
from task_models.action import Action
from task_models.task import HierarchicalTask, \
    LeafCombination, SequentialCombination, \
    ParallelCombination, AlternativeCombination

WRITE_TO_FILE = False
main_path = "/home/corina/sim_data"
path_sim_train = os.path.join(main_path, "train")
path_sim_test = os.path.join(main_path, "test")
path_sim_obj = "/home/corina/code/hrc_ws/src/hrteaming-bctask/" \
               "StateDiscretization/HMM/DevDebugCompleteDataSet/Features"


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

    def __init__(self, root=None, name=None, num_feats_action=None,
                 share_feats=False,
                 num_supp_bhvs=None, num_supp_bhvs_named=None, num_supp_bhvs_rd=None,
                 supp_bhvs_dict=dict(), supp_bhvs_named_dict=dict(), supp_bhvs_rd_dict=dict()):
        super(HierarchicalTaskHMM, self).__init__()
        self.root = root
        self.name = name
        self.all_trajectories = []
        self.bin_trajectories = []
        self.bin_trajectories_test = []
        self.obj_names = dict()
        self.obj_ints = dict()
        self.feat_names = dict()
        self.feat_ints = dict()
        self.num_obj = 0
        self.share_feats = share_feats
        self.num_feats_action = num_feats_action
        self.train_set_actions = []
        self.train_set_actions_traj = []
        self.test_set_actions = []
        self.test_set_actions_traj = []
        self.train_set_sb = []
        self.train_set_sb_actions = []
        self.train_set_sb_traj = []
        self.train_set_sb_blind_robot = []
        self.train_set_sb_actions_blind_robot = []
        self.train_set_sb_traj_blind_robot = []
        self.test_set_sb = []
        self.test_set_sb_traj = []
        self.test_set_sb_actions = []
        self.train_set_sb_named = []
        self.train_set_sb_actions_named = []
        self.train_set_sb_traj_named = []
        self.train_set_sb_blind_robot_named = []
        self.train_set_sb_actions_blind_robot_named = []
        self.train_set_sb_traj_blind_robot_named = []
        self.test_set_sb_named = []
        self.test_set_sb_traj_named = []
        self.test_set_sb_actions_named = []
        self.train_set_sb_rd = []
        self.train_set_sb_actions_rd = []
        self.train_set_sb_traj_rd = []
        self.train_set_sb_blind_robot_rd = []
        self.train_set_sb_actions_blind_robot_rd = []
        self.train_set_sb_traj_blind_robot_rd = []
        self.test_set_sb_rd = []
        self.test_set_sb_traj_rd = []
        self.test_set_sb_actions_rd = []
        self.train_set = []
        self.num_supp_bhvs = num_supp_bhvs
        self.user_prefs_dict = dict()
        self.supp_bhvs_dict = supp_bhvs_dict
        self.supp_bhvs_dict_rev = dict()
        self.num_supp_bhvs_named = num_supp_bhvs_named
        self.user_prefs_named_dict = dict()
        self.supp_bhvs_named_dict = supp_bhvs_named_dict
        self.supp_bhvs_named_dict_rev = dict()
        self.num_supp_bhvs_rd = num_supp_bhvs_rd
        self.user_prefs_rd_dict = dict()
        self.supp_bhvs_rd_dict = supp_bhvs_rd_dict
        self.supp_bhvs_rd_dict_rev = dict()
        self.gen_dict()

    @property
    def user_prefs_dict(self):
        return self.__user_prefs_dict

    @user_prefs_dict.setter
    def user_prefs_dict(self, user_prefs_dict):
        self.__user_prefs_dict = user_prefs_dict

    def gen_dict(self):
        regexp = r'( order)'
        self._gen_dict_rec(self.root, regexp)
        self.obj_ints = dict(map(reversed, self.obj_names.items()))
        if self.share_feats is True:
            for feat_idx in range(self.num_feats_action):
                self.feat_names["feat_" + str(feat_idx)] = feat_idx
        else:
            self.feat_names = self.obj_names
        self.feat_ints = dict(map(reversed, self.feat_names.items()))
        self.supp_bhvs_dict_rev = dict(map(reversed, self.supp_bhvs_dict.items()))
        self.supp_bhvs_named_dict_rev = dict(map(reversed, self.supp_bhvs_named_dict.items()))
        self.supp_bhvs_rd_dict_rev = dict(map(reversed, self.supp_bhvs_rd_dict.items()))

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

    def base_name(self, name):
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

    def write_task_dict_to_file(self, path, task_name):
        make_sure_path_exists(path)
        f = os.path.join(path, "task_{}_obj.txt".format(task_name))
        # if os.path.isfile(f) is False:
        f_out = open(f, 'w')
        if self.share_feats is False:
            sorted_vals = sorted(self.obj_names, key=self.obj_names.get)
            for el in sorted_vals:
                # f_out.write("{:>20} {}\n".format(el, obj_names[el]))
                f_out.write("{} {}\n".format(self.obj_names[el], el))
        else:
            for feat_idx in range(self.num_feats_action):
                f_out.write("{} {}\n".format(feat_idx, "feat_"+str(feat_idx)))
        f_out.close()
        # else:
        #     logging.warning("Did not save file task_{}_obj.txt "
        #                     "because it already exists. "
        #                     .format(task_name))

    def write_traj_to_file(self, path, bin_feats, task_name, traj_idx):
        make_sure_path_exists(path)
        f_out = os.path.join(path, "task_{}_traj_{}.bfout".
                             format(task_name, traj_idx))
        # if os.path.isfile(f_out) is False:
        np.savetxt(f_out, bin_feats, fmt=str("%d"))
        # else:
        #     logging.warning("Did not save file task_{}_traj_{}.bfout "
        #                     "because it already exists. "
        #                     "Continuing onto the next trajectory."
        #                     .format(task_name, traj_idx))

    def gen_bin_feats_test_set(self, path_train, path_test, task_name, test_set_size):
        for i in range(test_set_size):
            self.bin_trajectories_test.append(self.bin_trajectories[i])
            shutil.copy2(os.path.join(path_train, "task_{}_traj_{}.bfout".
                                      format(task_name, i)), path_test)
        self.bin_trajectories_test = np.array(self.bin_trajectories_test)

    def gen_bin_feats_traj(self, cum_feats=False):
        """Generates binary features for all possible trajectories from an HTM
        and writes them to file.
        Should only be called after calling gen_all_trajectories().
        """
        if self.all_trajectories:
            regexp = r'( order)'
            trajectories = list(zip(*self.all_trajectories))[1]
            bin_feats_init = np.array([0] * len(self.obj_names))
            task_name = "jdoe"
            if self.name:
                task_name = self.name
            #path = os.path.join(path_sim_train, task_name)
            if WRITE_TO_FILE is True:
                self.write_task_dict_to_file(path_sim_obj, task_name)
            path_train = os.path.join(path_sim_train, task_name)
            for traj_idx, traj in enumerate(trajectories):
                bin_feats = np.tile(bin_feats_init,
                                    (len(set(traj)) + 1, 1))
                for node_idx, node in enumerate(traj):
                    if cum_feats is True:
                        keys = [self.obj_names.get(key.name) if re.search(regexp, key.name) is None
                                else self.obj_names.get(key.name[:re.search(regexp, key.name).start()])
                                for key in traj[:node_idx + 1]]
                        bin_feats[node_idx + 1, keys] = 1
                    else:
                        action = self.obj_names.get(traj[node_idx].name)
                        if action is None:
                            action = self.obj_names.get(traj[node_idx]
                                                        .name[:re.search(regexp, traj[node_idx].name).start()])
                        bin_feats[node_idx + 1, action] = 1
                #bin_feats = unique_rows(bin_feats)
                self.bin_trajectories.append(bin_feats)
                if WRITE_TO_FILE is True:
                    self.write_traj_to_file(path_train, bin_feats, task_name, traj_idx)
            self.bin_trajectories = np.array(self.bin_trajectories)
            path_test = os.path.join(path_sim_test, task_name)
            if WRITE_TO_FILE is True:
                make_sure_path_exists(path_test)
                self.gen_bin_feats_test_set(path_train, path_test, task_name, 6)
                # for i in range(6):
                #     shutil.copy2(os.path.join(path, "task_{}_traj_{}.bfout".format(task_name, i)), path_test)
        else:
            raise ValueError("Cannot generate bin feats before generating all trajectories.")

    def gen_object_bin_feats_sb(self, num_feats, feat_probs):
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
        self.train_set_actions, self.train_set_actions_traj = \
            self._gen_set_bin_feats_actions(self.all_trajectories, size, path_sim_train, bias=bias)

    def gen_test_set_actions(self, size):
        if len(self.train_set_actions) == 0:
            raise ValueError("There exists no training set. Cannot create test set.")
        self.test_set_actions, self.test_set_actions_traj = \
            self._gen_set_bin_feats_actions(self.all_trajectories, size, path_sim_test)

    def _gen_set_bin_feats_actions(self, trajectories, size, path_sim_local,
                                   bias=False, bias_weight=50):
        """Generates sets to be used for a dataset only for actions.
        It excludes supportive behaviors."""
        len_traj = len(trajectories)
        set_bin_feats = []
        regexp = r'( order)'

        task_name = "jdoe"
        if self.name:
            task_name = self.name

        if WRITE_TO_FILE is True:
            self.write_task_dict_to_file(path_sim_obj, task_name)

        if self.share_feats is True:
            traj_bin_feats_init = np.array([0] * self.num_feats_action)
        else:
            traj_bin_feats_init = np.array([0] * (self.num_obj*self.num_feats_action))
        set_actions_traj_local = []
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
            set_actions_traj_local.append(traj)
            traj_bin_feats = np.tile(traj_bin_feats_init,
                                (len(set(traj)) + 1, 1))
            for node_idx, node in enumerate(traj):
                node_bin_feats_sb = self.gen_object_bin_feats_sb(
                    node.action.num_feats, node.action.feat_probs)
                if self.share_feats is True:
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
            path = os.path.join(path_sim_local, task_name)
            if WRITE_TO_FILE is True:
                self.write_traj_to_file(path, traj_bin_feats, task_name, i)
            set_bin_feats.append(traj_bin_feats)
        return np.array(set_bin_feats), set_actions_traj_local

    def gen_train_set_sb_blind_robot(self, user_prefs, test_set_size):
        self.train_set_sb_blind_robot, \
        self.train_set_sb_traj_blind_robot, \
        self.train_set_sb_actions_blind_robot = \
            self._gen_set_sb(self.train_set_actions_traj, self.train_set_actions,
                             user_prefs, test_set_size, sb_type='prob')

    def gen_train_set_sb_blind_robot_named(self, user_prefs, test_set_size):
        self.train_set_sb_blind_robot_named, \
        self.train_set_sb_traj_blind_robot_named, \
        self.train_set_sb_actions_blind_robot_named = \
            self._gen_set_sb(self.train_set_actions_traj, self.train_set_actions,
                             user_prefs, test_set_size, sb_type='named')

    def gen_train_set_sb_blind_robot_rd(self, user_prefs, test_set_size):
        self.train_set_sb_blind_robot_rd, \
        self.train_set_sb_traj_blind_robot_rd, \
        self.train_set_sb_actions_blind_robot_rd = \
            self._gen_set_sb(self.train_set_actions_traj, self.train_set_actions,
                             user_prefs, test_set_size, sb_type='rd')

    def gen_train_set_sb(self, user_prefs, num_dems):
        self.train_set_sb, self.train_set_sb_traj, self.train_set_sb_actions = \
            self._gen_set_sb(self.train_set_actions_traj, self.train_set_actions,
                             user_prefs, num_dems, sb_type='prob')

    def gen_test_set_sb(self, user_prefs, num_dems):
        self.test_set_sb, self.test_set_sb_traj, self.test_set_sb_actions = \
            self._gen_set_sb(self.test_set_actions_traj, self.test_set_actions, user_prefs,
                             num_dems, sb_type='prob')

    def gen_train_set_sb_named(self, user_prefs, num_dems):
        self.train_set_sb_named, self.train_set_sb_traj_named, self.train_set_sb_actions_named = \
            self._gen_set_sb(self.train_set_actions_traj, self.train_set_actions,
                             user_prefs, num_dems, sb_type='named')

    def gen_test_set_sb_named(self, user_prefs, num_dems):
        self.test_set_sb_named, self.test_set_sb_traj_named, self.test_set_sb_actions_named = \
            self._gen_set_sb(self.test_set_actions_traj, self.test_set_actions,
                             user_prefs, num_dems, sb_type='named')

    def gen_train_set_sb_rd(self, user_prefs, num_dems):
        self.train_set_sb_rd, self.train_set_sb_traj_rd, self.train_set_sb_actions_rd = \
            self._gen_set_sb(self.train_set_actions_traj, self.train_set_actions,
                             user_prefs, num_dems, sb_type='rd')

    def gen_test_set_sb_rd(self, user_prefs, num_dems):
        self.test_set_sb_rd, self.test_set_sb_traj_rd, self.test_set_sb_actions_rd = \
            self._gen_set_sb(self.test_set_actions_traj, self.test_set_actions,
                             user_prefs, num_dems, sb_type='rd')

    def _gen_set_sb(self, trajectories, traj_actions, users_prefs, num_dems, sb_type='rd'):
        len_train_traj = len(trajectories)
        if len_train_traj == 0:
            raise ValueError("Please generate training set w/ actions "
                             "before generating training supportive behavior set.")
        set_sb = []
        set_sb_traj = []
        set_sb_actions = []
        for user in sorted(users_prefs.keys()):
            if sb_type == 'prob':
                all_dems_user_bin_feats, user_sb_traj, user_sb_actions = \
                    self._gen_user_sb(trajectories, traj_actions, users_prefs[user], num_dems)
            elif sb_type == 'named':
                all_dems_user_bin_feats, user_sb_traj, user_sb_actions = \
                    self._gen_user_sb_named(trajectories, traj_actions, users_prefs[user], num_dems)
            elif sb_type == 'rd':
                all_dems_user_bin_feats, user_sb_traj, user_sb_actions = \
                    self._gen_user_sb_rd(trajectories, traj_actions, users_prefs[user], num_dems)
            set_sb.append(all_dems_user_bin_feats)
            set_sb_traj.append(user_sb_traj)
            set_sb_actions.append(user_sb_actions)
        return np.array(set_sb), np.array(set_sb_traj), np.array(set_sb_actions)

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
            traj_user_bin_feats.insert(0, np.array([0] * self.num_supp_bhvs))
            all_dems_user_bin_feats.append(traj_user_bin_feats)
        return all_dems_user_bin_feats, user_sb_traj, user_sb_actions

    def _gen_user_sb_named(self, trajectories, traj_actions, user_prefs, num_dems):
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
            traj_user_bin_feats = np.zeros((len_traj, self.num_supp_bhvs_named))
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
                        traj_user_bin_feats[leaf_idx+1, self.supp_bhvs_named_dict[key]] = 1
            all_dems_user_bin_feats.append(traj_user_bin_feats)
        return all_dems_user_bin_feats, user_sb_traj, user_sb_actions

    def _gen_user_sb_rd(self, trajectories, traj_actions, user_prefs, num_dems):
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
            traj_user_bin_feats = np.zeros((len_traj, self.num_supp_bhvs_rd))
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
                        traj_user_bin_feats[leaf_idx+1, self.supp_bhvs_rd_dict[key]] = 1
            all_dems_user_bin_feats.append(traj_user_bin_feats)
        return all_dems_user_bin_feats, user_sb_traj, user_sb_actions

    def reset_sb_sets(self):
        self.train_set_sb = []
        self.train_set_sb_actions = []
        self.train_set_sb_traj = []
        self.test_set_sb = []
        self.test_set_sb_traj = []
        self.test_set_sb_actions = []

    def reset_sb_sets_named(self):
        self.train_set_sb_named = []
        self.train_set_sb_actions_named = []
        self.train_set_sb_traj_named = []
        self.test_set_sb_named = []
        self.test_set_sb_traj_named = []
        self.test_set_sb_actions_named = []

    def reset_sb_sets_rd(self):
        self.train_set_sb_rd = []
        self.train_set_sb_actions_rd = []
        self.train_set_sb_traj_rd = []
        self.test_set_sb_rd = []
        self.test_set_sb_traj_rd = []
        self.test_set_sb_actions_rd = []

    def reset_sb_sets_blind_robot(self):
        self.train_set_sb_blind_robot = []
        self.train_set_sb_actions_blind_robot = []
        self.train_set_sb_traj_blind_robot = []
        self.train_set_sb_blind_robot_named = []
        self.train_set_sb_actions_blind_robot_named = []
        self.train_set_sb_traj_blind_robot_named = []
        self.train_set_sb_blind_robot_rd = []
        self.train_set_sb_actions_blind_robot_rd = []
        self.train_set_sb_traj_blind_robot_rd = []

    def reset_main_sets(self):
        self.all_trajectories = []
        self.bin_trajectories = []
        self.bin_trajectories_test = []
        self.train_set_actions = []
        self.train_set_actions_traj = []
        self.test_set_actions = []
        self.test_set_actions_traj = []