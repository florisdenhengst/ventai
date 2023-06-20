import copy
from collections.abc import Iterable
import functools
import itertools
import operator
from matplotlib import pyplot as plt
import argparse

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import numpy_ext as npe
import math
import random
from pprint import pprint
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.sparse import hstack, vstack, csr_matrix
import scipy

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import Normalizer, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import importlib
import concurrent
import time
import traceback

import seaborn as sns

import utils
import safety
import ope

import sys

from config import demographics, vital_sign_vars, lab_vars, treatment_vars, vent_vars, guideline_vars, ffill_windows_clinical, SAMPLE_TIME_H
from config import fio2_bins, peep_bins, tv_bins

parser = argparse.ArgumentParser(
                    prog='ope_script',
                    description='Runs OPE for single experiment (seed, unsafety_prob, shaping)')

parser.add_argument('seed', type=int, nargs=1)
parser.add_argument('shaping', choices=['avgpotential2', 'allpotential', 'avgbase', 'allbase', 'none'], default='none', nargs=1)
parser.add_argument('unsafety_prob', type=float, nargs=1)
parser.add_argument('shaping_scalar', type=float, nargs='?', default=0.0)
parser.add_argument('train_test', choices=['train', 'test'], default='test', nargs='?')

args = parser.parse_args()

seed = args.seed[0]
shaping = args.shaping[0]
unsafety_prob = args.unsafety_prob[0]
train_test = args.train_test
shaping_scalar = args.shaping_scalar

if shaping_scalar == 0.0 and shaping != 'none':
    raise ValueError('Shaping scalar is 0.0 while shaping is {}'.format(shaping))

# TODO FdH: remove SOLOTEST variable and its uses
SOLOTEST = False

test_set_file = 'data/test_unshaped_traj_{}.csv'
train_set_file = 'data/train_unshaped_traj_{}.csv'

#models/mcp_<ACTION_SELECTION_<SEED>_<SHAPING_NAME>_<SHAPING_SCALAR>_<UNSAFETY_PROB>.bin
greedy_policy_file = 'models/mcp_greedy_policy_{}_{}_{}_{}.bin'
sm_policy_file = 'models/mcp_softmax_policy_{}_{}_{}_{}.bin'
behavior_policy_train_file = 'models/clinicians_policy_train_{}{}.bin'
behavior_policy_test_file = 'models/clinicians_policy_test_{}{}.bin'
behavior_policy_file = 'models/clinicians_policy_train_test_{}{}.bin'

TERMINAL_MORT = 650
TERMINAL_NONMORT = 651

def add_state_compl(dataset, aggregator=np.mean):
    dataset[:, 'compl'] = safety.compliance_clinical(

def add_traj_return(dataset):
    return_set = dataset.copy()
    return_set['traj_reward'] = np.nan
    return_set.loc[return_set.mort90day == 't', 'traj_reward'] = -100
    return_set.loc[return_set.mort90day == 'f', 'traj_reward'] = 100
    return_set['traj_return'] = (.99 ** return_set['traj_len']) * return_set['traj_reward']
    return return_set

def add_scaled_traj_return(dataset):
    return_set = dataset.copy()
    return_set['traj_reward'] = np.nan
    return_set.loc[return_set.mort90day == 't', 'traj_reward'] = 0
    return_set.loc[return_set.mort90day == 'f', 'traj_reward'] = 1
    return_set['traj_return'] = (.99 ** return_set['traj_len']) * return_set['traj_reward']
    return return_set

def add_traj_len(dataset):
    assert dataset.traj_count.isna().sum() == 0
    return_set = dataset.copy()
    return_set['traj_len'] = return_set.groupby('icustay_id')['traj_count'].transform('max')
    return_set['traj_len'] = return_set['traj_len'] + 1
    return return_set

def fix_next_terminal_state(dataset):
    if dataset[dataset.terminal].next_state.nunique() == 2:
        # no fix necessary
        return dataset
    else:
        return_set = dataset.copy()
        return_set.loc[return_set.terminal & (return_set.mort90day == 't'), 'next_state'] = TERMINAL_MORT
        return_set.loc[return_set.terminal & (return_set.mort90day == 'f'), 'next_state'] = TERMINAL_NONMORT
        return return_set

def postprocess(dataset):
    ds = fix_next_terminal_state(add_traj_return(add_traj_len(dataset)))
    ds.next_state = ds.next_state.astype('int')
    ds.reward = ds.reward.astype('float')
    return ds

results = []
test_set = postprocess(pd.read_csv(test_set_file.format(seed)))
train_set = postprocess(pd.read_csv(train_set_file.format(seed), low_memory=False))
train_test_set = pd.concat([train_set, test_set])

behavior_policy = joblib.load(behavior_policy_file.format(seed,''))
behavior_train_policy = joblib.load(behavior_policy_train_file.format(seed,''))
n_states, n_actions = behavior_policy.shape
behavior_train_policy = utils.repair_policy(
            behavior_train_policy,
            behavior_policy)
assert np.absolute(1.0 - behavior_train_policy.sum(axis=1) <1e-10).all(), "behavior_train policy does not sum to 1.0 somewhere"

            
behavior_test_policy = joblib.load(behavior_policy_test_file.format(seed, ''))
behavior_safe_train = safety.repaired_safe(behavior_train_policy, behavior_train_policy)
    
np.random.seed(seed)
if 'none' in shaping:
    shaped = False
else:
    shaped = True

assert test_set.traj_reward.isna().sum() == 0, "Zero rewards in test set"
assert train_set.traj_reward.isna().sum() == 0, "Zero rewards in train set "

#         greedy_unsafe = utils.repair_unsupported_greedy_policy(
#             joblib.load(greedy_policy_file.format(seed, shaping_fname[shaping])),
#             train_set
#         )
#         greedy_safe = safety.repaired_safe(greedy_unsafe, behavior_train_policy, greedy=True)
sm_unsafe = joblib.load(sm_policy_file.format(seed, shaping, shaping_scalar, unsafety_prob))
fallback_sm_unsafe = np.tile(np.nanmean(behavior_train_policy, axis=0), n_states).reshape(n_states, n_actions) # avg policy across all states
sm_unsafe = utils.normalize_policy_probs(
        utils.repair_policy(
            sm_unsafe,
            behavior_train_policy))
assert np.absolute(1.0 - sm_unsafe.sum(axis=1) <1e-10).all(), "sm_unsafe policy does not sum to 1.0 somewhere"
if unsafety_prob == 1.0:
    sm_safe = safety.repaired_safe(sm_unsafe, behavior_train_policy)


if train_test == 'test':
    evaluations = [
    #             (train_set, greedy_unsafe, behavior_policy, 'train', 'greedy', shaped, shaping_scalar, 'unsafe', seed),
    #             (test_set, greedy_unsafe, behavior_policy, 'test', 'greedy', shaped, shaping_scalar, 'unsafe', seed),
    #             (train_set, greedy_safe, behavior_policy, 'train', 'greedy', shaped, shaping_scalar, 'safe', seed),
    #             (test_set, greedy_safe, behavior_policy, 'test', 'greedy', shaped, shaping_scalar, 'safe', seed),
    #    (train_set, sm_unsafe, behavior_policy, 'train', 'softmax', shaped, shaping, shaping_scalar, 'unsafe', unsafety_prob, seed),
        (test_set, sm_unsafe, behavior_policy, 'test', 'softmax', shaped, shaping, shaping_scalar, 'unsafe', unsafety_prob, seed),
    ]
else:
    evaluations += [
            (train_set, sm_unsafe, behavior_policy, 'train', 'softmax', shaped, shaping, shaping_scalar, 'unsafe', unsafety_prob, seed),
            ]

if unsafety_prob == 1.0 and not SOLOTEST:
    if train_test == 'test':
        evaluations += [
                (test_set, sm_safe, behavior_policy, 'test', 'softmax', shaped, shaping, shaping_scalar, 'safe', unsafety_prob, seed),
            ]
    else:
        evaluations += [
                (train_set, sm_safe, behavior_policy, 'train', 'softmax', shaped, shaping,  shaping_scalar, 'safe', unsafety_prob, seed),
                ]
#
if shaping == 'none':
    if train_test == 'test' and unsafety_prob == 1.0 and not SOLOTEST:
        evaluations += [
            (test_set, behavior_test_policy, behavior_test_policy, 'test', 'observed', shaped, shaping, shaping_scalar, 'unsafe', unsafety_prob, seed),
            (test_set, behavior_train_policy, behavior_policy, 'test', 'behavior', shaped, shaping, shaping_scalar, 'unsafe', unsafety_prob, seed),
            (test_set, behavior_safe_train, behavior_policy, 'test', 'behavior', shaped, shaping, shaping_scalar, 'safe', unsafety_prob, seed),
        ]
    elif train_test == 'train' and unsafety_prob == 1.0:
        evaluations += [
                (train_set, behavior_train_policy, behavior_train_policy, 'train', 'observed', shaped, shaping, shaping_scalar, 'unsafe', unsafety_prob, seed),
                (train_set, behavior_train_policy, behavior_policy, 'train', 'behavior', shaped, shaping, shaping_scalar, 'unsafe', unsafety_prob, seed),
                ]

for ds, evaluation_policy, behavior_policy, *config in evaluations:
    wis_mean, var, traj_weights = ope.wis_policy(ds, evaluation_policy, behavior_policy)
    q_estimator, v_estimator = ope.infer_estimators_func(train_test_set, evaluation_policy, 0.99, 50)
    phwis_mean, var, _ = ope.phwis_policy(ds, evaluation_policy, behavior_policy)
    fqe, wdr_mean = ope.wdr_policy(ds, evaluation_policy, behavior_policy, q_estimator, v_estimator, 0.99)
    phwdr_mean = ope.phwdr_policy(ds, evaluation_policy, behavior_policy, q_estimator, v_estimator, 0.99)
    assert not np.isnan(phwdr_mean), "phwdr mean is nan!"
#             am = ope.am(ds, evaluation_policy, behavior_policy, delta=0.05)
#             hcope5 = ope.hcope(ds, evaluation_policy, behavior_policy, delta=0.05, c=5)
    am, hcope5 = np.nan, np.nan
    ess = ope.ess(traj_weights)
    results.append(list(map(str, (*config, wis_mean, phwis_mean, wdr_mean, phwdr_mean, fqe, ess, var, am, hcope5, len(train_set), len(test_set), train_set.icustay_id.nunique(), test_set.icustay_id.nunique()))))

for experiment in results:
    print(','.join(map(str, experiment)))
