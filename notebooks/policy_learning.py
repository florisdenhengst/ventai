#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import copy
from collections.abc import Iterable
import functools
import itertools
import operator
from matplotlib import pyplot as plt

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

import seaborn as sns

import argparse
import sys
import os

from config import demographics, vital_sign_vars, lab_vars, treatment_vars, vent_vars, guideline_vars, ffill_windows_clinical, SAMPLE_TIME_H
from config import fio2_bins, peep_bins, tv_bins
import safety
import utils

parser = argparse.ArgumentParser(
                    prog='policy_learning',
                    description='Creates Q table and policies')
parser.add_argument('seed', type=int, nargs=1)
parser.add_argument('shaping', choices=['avgpotential2', 'allpotential', 'avgbase', 'allbase', 'none'], default='none', nargs=1)
parser.add_argument('compliance_scalar', type=float, nargs=1)
parser.add_argument('unsafety_prob', type=float, nargs=1)
parser.add_argument('softmax_temp', type=float, nargs='?', default=1.0)
parser.add_argument('plot', type=bool, nargs='?', default=False)

args = parser.parse_args()

seed = args.seed[0]
SHAPING = args.shaping[0]
COMPLIANCE_SCALAR = args.compliance_scalar[0]
UNSAFETY_PROB = args.unsafety_prob[0] # should be in [0.0,1.0] or {0,0, 1.0} until safety probs implemented
SOFTMAX_TEMPERATURE = args.softmax_temp
plot = args.plot


# In[28]:


data_dir = 'data/'

GAMMA = 0.99
N_EPOCHS = 10000
LEARNING_RATE = 0.01

n_states = 650

if SHAPING == 'none' and COMPLIANCE_SCALAR != 0.0:
    raise Error('COMPLIANCE_SCALAR should be 0.0 if shaping approach is none')
elif SHAPING != 'none' and COMPLIANCE_SCALAR <= 0:
    raise Error('COMPLIANCE_SCALAR of {} not allowed for shaping approach {}, should be >0'.format(COMPLIANCE_SCALAR, SHAPING))


# In[6]:


# TODO FdH: import trajectories -- do some simple checks
test_set_file = 'data/test_unshaped_traj_{}.csv'
train_set_file = 'data/train_unshaped_traj_{}.csv'

train_set = pd.read_csv(train_set_file.format(seed))
test_set = pd.read_csv(test_set_file.format(seed))


# In[10]:


def compliance_to_potential(compliance):
    return compliance * COMPLIANCE_SCALAR

def potential_diff(x):
    if np.isnan(x.iloc[1]):
        p1 = 0.0 # see Grzes, AAMAS 2017
    else:
        p1 = GAMMA * x.iloc[1]
    return p1 - x.iloc[0]

# TODO FdH: reward shaping
if SHAPING == 'avgpotential2' or SHAPING == 'avgbase':
    train_set['compliance'] = safety.state_compliance_clinical(train_set, safety.avg_clinical_timestep)
elif SHAPING == 'allpotential' or SHAPING == 'allbase':
    train_set['compliance'] = safety.state_compliance_clinical(train_set, safety.all_clinical_timestep)
elif SHAPING == 'none':
    train_set['compliance'] = 0.0
else:
    raise ValueError('Unknown shaping approach')
train_set['potential'] = compliance_to_potential(train_set['compliance'])
if 'potential' in SHAPING:
    train_set['shaping_reward_unshift'] = train_set.groupby('icustay_id').rolling(window=2)['potential'].apply(potential_diff).fillna(0.0).reset_index().set_index('level_1')['potential']
    train_set['shaping_reward'] = train_set['shaping_reward_unshift'].shift(-1)
    train_set.loc[train_set.terminal, 'shaping_reward'] = train_set['potential']
elif 'base' in SHAPING:
    train_set['shaping_reward'] = train_set['potential']
elif SHAPING == 'none':
    train_set['shaping_reward'] = 0.0

train_set['reward'] = train_set.reward + train_set.shaping_reward


# In[13]:


# According to the tabular FQI algorithm in Ernst, Geurts & Wehenkel (2005), Figure 1
# and Peine's supplementary discussion "A: Evaluation of Policies".
def peine_mc_iterate(snsasr, Qn, gamma, n_epochs=1, learning_rate=0.1, unsafety_prob=0.0, safety_map=safety.action_id_compliance):
    """
    Monte-carlo-based iteration of the training procedure according to tabular FQI & Peine's supplementary discussion.
    
    snsas: numpy ndarray with discretized state-nextstate-action tuples
    r: a function that returns the immediate reward for a state-action pair
    Qn: dictionary that maps iteration indices to Qn-estimates
    n: iteration number
    gamma: discount factor
    n_epochs: number of times to iterate over dataset
    learning rate: learning rate alpha
    """
    def epoch(snsasr, Qn, gamma, learning_rate, unsafety_prob, safety_map):
        for i, (s, ns, a, er) in enumerate(snsasr):
            if unsafety_prob == 1.0:
                # We do not care about the safety rules
                Qn[s,a] = Qn[s,a] + learning_rate * (er + gamma * np.max(Qn[int(ns),:]) - Qn[s,a])
            elif unsafety_prob == 0.0:
                if safety_map[a]:
                    Qn[s,a] = Qn[s,a] + learning_rate * (er + gamma * np.max(Qn[int(ns), safety_map]) - Qn[s,a])
                else:
                    # taken action not safe, disregard sample
                    pass
            else:
                raise ValueError("Only unsafety probs in {0.0, 1.0} supported for now")
                #TODO FdH: implement unsafety probs (0.0, 0.0}
        return Qn
    assert Qn.shape == (n_states+2, 7**3)
    assert safety_map is not None or unsafety_prob == 0.0
    for n in range(n_epochs):
        Qn = epoch(snsasr, Qn, gamma, learning_rate, unsafety_prob, safety_map)
        #assert np.nanmax(Qn) < 100, "Scores > 100 should not occur, found: {}".format(np.nanargmax(Qn))
        print('.', end='')
    return Qn


# ## Q learning
# Derive a table of Q values

# In[16]:


# learn a q table
q_init_val = 0
q_init = np.full((n_states + 2, 7**3), float(q_init_val))
#  peine_mc_iterate(snsas, r, Qn, gamma, n_epochs=1, learning_rate=0.1):
q_mcp = peine_mc_iterate(
    # TODO: why are the NaNs here? how to deal with these?
    snsasr=train_set[['state', 'next_state', 'action_discrete', 'reward']].astype(int).to_numpy(),
    Qn=q_init,
    gamma=GAMMA,
    n_epochs=N_EPOCHS,
    learning_rate=LEARNING_RATE,
    unsafety_prob=UNSAFETY_PROB,
    safety_map=safety.action_id_compliance
)


# In[18]:


# postprocess q table -- removal of nans
q_mcp_nan = q_mcp.copy()[:n_states, :]
q_mcp_nan[q_mcp_nan == 0.0] = np.nan
joblib.dump(
    {'hyperparameters': {
        'Q_init': q_init,
        'gamma': GAMMA,
        'n_epochs': N_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'shaping': SHAPING,
        'shaping_scalar': COMPLIANCE_SCALAR,
        'unsafety_prob':UNSAFETY_PROB,
        'softmax_temp': SOFTMAX_TEMPERATURE,
        'safety_map':safety.action_id_compliance            
    },
    'model': q_mcp_nan,
    },
    'models2/peine_mc_{}_{}_{}_q_table_{}.bin'.format(SHAPING, UNSAFETY_PROB, SOFTMAX_TEMPERATURE, seed),
    compress=True
)


# ## Policy learning
# Derive a policy from a table of Q values

# In[19]:


# derive policy by taking argmax over non-nan q values
q_mcp_nan[np.isnan(q_mcp_nan).all(axis=1),:] = 0

best_action_indices = np.nanargmax(q_mcp_nan, axis=1)

action_index_grid = np.tile(np.array(range(7**3)), 650).reshape((650, 7**3))
best_action_grid = np.repeat(best_action_indices, 7**3).reshape((650, 7**3))
best_action_bool = best_action_grid == action_index_grid
assert best_action_bool.shape == (n_states, 7**3)
assert (best_action_bool.sum(axis=1) == 1).all()
mcp_greedy = best_action_bool.astype(float)
assert (mcp_greedy.sum(axis=1) == 1).all()

# derive policy by taking softmax
q_mcp_neg = q_mcp.copy()[:n_states, :]
q_mcp_neg[q_mcp_neg == 0.0] = float('-inf')
mcp_softmax = scipy.special.softmax(q_mcp_neg / SOFTMAX_TEMPERATURE, axis=1)
assert mcp_softmax.shape == (n_states, 7**3)
assert (mcp_greedy.sum(axis=1) == 1).all()


# In[20]:


# some diagnostics
best_s, best_a = np.unravel_index(np.nanargmax(q_mcp_nan), (n_states, 7**3))
print("Global highest Q value {} for tv, fio2, peep ranges: {}".format(q_mcp_nan[best_s, best_a], utils.to_action_ranges(best_a)))
best_mean_a, best_mean_a_q = np.nanargmax(np.nanmean(q_mcp_nan, axis=0)), np.nanmax(np.nanmean(q_mcp_nan, axis=0))
print("Highest avg Q value across states {} for tv, fio2, peep ranges: {}".format(best_mean_a_q, utils.to_action_ranges(best_mean_a)))
best_med_a, best_med_a_q = np.nanargmax(np.nanmedian(q_mcp_nan, axis=0)), np.nanmax(np.nanmedian(q_mcp_nan, axis=0))
print("Highest median Q value across states {} for tv, fio2, peep ranges: {}".format(best_med_a, utils.to_action_ranges(best_med_a)))


# In[21]:


def sortnan(x, index):
    return float('-inf') if np.isnan(x[index]) else x[index]

if plot:
    sns.histplot(q_mcp_nan[mcp_greedy == 1.0].ravel(), log_scale=(False, True), bins=200)
    plt.xlabel('Q value')
    plt.title('Histogram of Q values greedy policy')
    plt.show()
    train_set['positive_outcome'] = (train_set['mort90day'] == 't') | (train_set['hospmort'] == 'f')
    estimated_mort_state_visit = train_set.groupby('state').mean('positive_outcome')[['positive_outcome']].to_numpy()
    sns.scatterplot(x=np.nanmean(q_mcp_nan, axis=1), y=estimated_mort_state_visit.reshape(n_states,))
    plt.xlabel('Mean estimated Q value')
    plt.ylabel('Average outcome')
    plt.title('Outcome vs mean Q value estimates')
    plt.show()
    sns.scatterplot(x=np.nanmax(q_mcp_nan, axis=1), y=estimated_mort_state_visit.reshape(n_states,))
    plt.xlabel('Max estimated Q value')
    plt.ylabel('Average outcome')
    plt.title('Outcome vs max Q value estimates')
    plt.show()
    sns.scatterplot(x=np.nanmedian(q_mcp_nan, axis=1), y=estimated_mort_state_visit.reshape(n_states,))
    plt.xlabel('Max estimated Q value')
    plt.ylabel('Average outcome')
    plt.title('Outcome vs median Q value estimates')
    plt.show()
    
    q_vars = np.nanvar(q_mcp_nan, axis=1)
    q_means = np.nanmean(q_mcp_nan, axis=1)
    q_medians = np.nanmedian(q_mcp_nan, axis=1)
    q_maxs = np.nanmax(q_mcp_nan, axis=1)
    q_mins = np.nanmin(q_mcp_nan, axis=1)
    stacked = np.column_stack((q_means, q_medians, q_maxs, q_mins, q_vars))
    xs = range(n_states)
    means_sorted = np.array(sorted(stacked, key=lambda x: x[0]))
    means_upper = means_sorted[:, 0] + means_sorted[:, -1]
    means_lower = means_sorted[:, 0] - means_sorted[:, -1]
    axs = sns.lineplot(x=xs, y=means_sorted[:, 0])
    axs.fill_between(x=xs, y1=means_lower, y2=means_upper, alpha=.3)
    axs.set_ylim(-150, 150)
    plt.xlabel('State')
    plt.ylabel('Q value')
    plt.title('Mean Q values per state +- 1 var')
    plt.show()

    medians_sorted = np.array(sorted(stacked, key=lambda x: x[1]))
    means_upper = medians_sorted[:, 0] + medians_sorted[:, -1]
    means_lower = medians_sorted[:, 0] - medians_sorted[:, -1]
    axs = sns.lineplot(x=xs, y=medians_sorted[:, 1])
    axs.fill_between(x=xs, y1=means_lower, y2=means_upper, alpha=.3)
    axs.set_ylim(-150, 150)
    plt.xlabel('State')
    plt.ylabel('Q value')
    plt.title('Median Q values per state +- 1 var')
    plt.show()

    mins_sorted = np.array(sorted(stacked, key=lambda x: x[3]))
    axs = sns.scatterplot(x=xs, y=mins_sorted[:, 3], color='orange', alpha=.5)
    plt.xlabel('State')
    plt.ylabel('Q value')
    plt.title('Min Q values per state')

    maxs_sorted = np.array(sorted(stacked, key=lambda x: x[2]))
    axs = sns.lineplot(x=xs, y=maxs_sorted[:, 2], alpha=.5)
    plt.xlabel('State')
    plt.ylabel('Q value')
    plt.title('Max Q values per state')
    plt.show()

    q_vars = np.nanvar(q_mcp_nan, axis=0)
    q_means = np.nanmean(q_mcp_nan, axis=0)
    q_medians = np.nanmedian(q_mcp_nan, axis=0)
    q_maxs = np.nanmax(q_mcp_nan, axis=0)
    q_mins = np.nanmin(q_mcp_nan, axis=0)
    stacked = np.column_stack((q_means, q_medians, q_maxs, q_mins, q_vars))
    xs = range(7**3)
    means_sorted = np.array(sorted(stacked, key=lambda x: sortnan(x, 0)))
    means_upper = means_sorted[:, 0] + means_sorted[:, -1]
    means_lower = means_sorted[:, 0] - means_sorted[:, -1]
    axs = sns.lineplot(x=xs, y=means_sorted[:, 0])
    axs.fill_between(x=xs, y1=means_lower, y2=means_upper, alpha=.3)
    axs.set_ylim(-150, 150)
    plt.xlabel('State')
    plt.ylabel('Q value')
    plt.title('Mean Q values per action +- 1 var')
    plt.show()

    medians_sorted = np.array(sorted(stacked, key=lambda x: sortnan(x, 1)))
    means_upper = medians_sorted[:, 0] + medians_sorted[:, -1]
    means_lower = medians_sorted[:, 0] - medians_sorted[:, -1]
    axs = sns.lineplot(x=xs, y=medians_sorted[:, 1])
    axs.fill_between(x=xs, y1=means_lower, y2=means_upper, alpha=.3)
    axs.set_ylim(-150, 150)
    plt.xlabel('State')
    plt.ylabel('Q value')
    plt.title('Median Q values per action +- 1 var')
    plt.show()

    maxs_sorted = np.array(sorted(stacked, key=lambda x: sortnan(x, 2)))
    axs = sns.lineplot(x=xs, y=maxs_sorted[:, 2])
    plt.xlabel('State')
    plt.ylabel('Q value')
    plt.title('Max Q values per action')
    plt.show()

    mins_sorted = np.array(sorted(stacked, key=lambda x: sortnan(x, 3)))
    axs = sns.lineplot(x=xs, y=mins_sorted[:, 3])
    plt.xlabel('State')
    plt.ylabel('Q value')
    plt.title('Min Q values per action')
    plt.show()


# ## Derive behavior policies

# In[22]:


if UNSAFETY_PROB == 1.0 and SHAPING == 'none': # only derive behavior policy if all actions are allowed
    # Train set
    behavior_policy_df = (test_set.value_counts(['state', 'action_discrete']) / test_set.value_counts(['state']))
    assert (1.0 - behavior_policy_df.groupby('state').sum() < 1e10).all(), "Policy action probs should sum to 1 per state"
    behavior_policy_df = behavior_policy_df.reset_index()

    behavior_policy_df = train_set.value_counts(['state', 'action_discrete']) / train_set.value_counts(['state'])
    assert (1.0 - behavior_policy_df.groupby('state').sum() < 1e10).all(), "Policy action probs should sum to 1 per state"

    behavior_policy_pivot = behavior_policy_df.reset_index().pivot(columns='action_discrete', index='state')[0]
    behavior_policy_states = set(behavior_policy_pivot.index.unique())
    for s in range(n_states):
        if s not in behavior_policy_states:
            action_probs = [1.0 / (7**3),] * 7**3 # uniform distribution
            for i, p in enumerate(action_probs):
                behavior_policy_pivot.loc[s] = [s, i, p]

    behavior_policy_pivot = behavior_policy_pivot.sort_values(['state'])

    for a in range(7**3):
        if a not in behavior_policy_pivot.columns:
            behavior_policy_pivot.loc[:, a] = np.nan

    behavior_policy_nan = behavior_policy_pivot[range(7**3)].to_numpy()
    assert (1- (np.nansum(behavior_policy_nan, axis=1)) < 1e10).all(), "Policy action probs should sum to 1 per state"
    behavior_policy = np.nan_to_num(behavior_policy_nan, 0.0)
    assert (1- (behavior_policy.sum(axis=1)) < 1e10).all(), "Policy action probs should sum to 1 per state"
    assert behavior_policy.shape == (n_states, 7**3), "Behavior policy should cover all states and actions"
    mcp_greedy_mask = mcp_greedy.astype(bool)
    assert (mcp_greedy_mask.sum(axis=1) == 1).all(), "Greedy policy mask should mask out all-but-one action"
    joblib.dump(behavior_policy, "models2/clinicians_policy_train_{}.bin".format(seed), compress=True)
    
    # Test set
    behavior_policy_df = test_set.value_counts(['state', 'action_discrete']) / test_set.value_counts(['state'])
    assert (1.0 - behavior_policy_df.groupby('state').sum() < 1e10).all(), "Policy action probs should sum to 1 per state"

    behavior_policy_pivot = behavior_policy_df.reset_index().pivot(columns='action_discrete', index='state')[0]

    for a in range(7**3):
        if a not in behavior_policy_pivot.columns:
            behavior_policy_pivot.loc[:, a] = np.nan
    behavior_policy_states = set(behavior_policy_pivot.index.unique())
    for s in range(n_states):
        if s not in behavior_policy_states:
            action_probs = [1.0 / (7**3),] * 7**3 # uniform distribution
            for i, p in enumerate(action_probs):
                behavior_policy_pivot.loc[s] = [1/(7**3),]*(7**3)

    behavior_policy_pivot = behavior_policy_pivot.sort_values(['state'])

    # train + test set
    behavior_policy_nan = behavior_policy_pivot[range(7**3)].to_numpy()
    assert (1- (np.nansum(behavior_policy_nan, axis=1)) < 1e10).all(), "Policy action probs should sum to 1 per state"
    behavior_policy = np.nan_to_num(behavior_policy_nan, 0.0)
    assert (1- (behavior_policy.sum(axis=1)) < 1e10).all(), "Policy action probs should sum to 1 per state"
    assert behavior_policy.shape == (n_states, 7**3), "Behavior policy should cover all states and actions"
    mcp_greedy_mask = mcp_greedy.astype(bool)
    assert (mcp_greedy_mask.sum(axis=1) == 1).all(), "Greedy policy mask should mask out all-but-one action"
    joblib.dump(behavior_policy, "models2/clinicians_policy_test_{}.bin".format(seed), compress=True)
    
    train_test = pd.concat([train_set, test_set])
    behavior_policy_df = train_test.value_counts(['state', 'action_discrete']) / train_test.value_counts(['state'])
    assert (1.0 - behavior_policy_df.groupby('state').sum() < 1e10).all(), "Policy action probs should sum to 1 per state"

    behavior_policy_pivot = behavior_policy_df.reset_index().pivot(columns='action_discrete', index='state')[0]
    for a in range(7**3):
        if a not in behavior_policy_pivot.columns:
            behavior_policy_pivot.loc[:, a] = np.nan

    behavior_policy_nan = behavior_policy_pivot[range(7**3)].to_numpy()
    assert (1- (np.nansum(behavior_policy_nan, axis=1)) < 1e10).all(), "Policy action probs should sum to 1 per state"
    behavior_policy = np.nan_to_num(behavior_policy_nan, 0.0)
    assert (1- (behavior_policy.sum(axis=1)) < 1e10).all(), "Policy action probs should sum to 1 per state"
    assert behavior_policy.shape == (n_states, 7**3), "Behavior policy should cover all states and actions"
    mcp_greedy_mask = mcp_greedy.astype(bool)
    assert (mcp_greedy_mask.sum(axis=1) == 1).all(), "Greedy policy mask should mask out all-but-one action"
    joblib.dump(behavior_policy, "models2/clinicians_policy_train_test_{}.bin".format(seed), compress=True)
    print("Entropy train-test behavior policy: {}".format(scipy.stats.entropy(behavior_policy.ravel())))
    print("Behavior policy argmax and greedy policy agreement: {}".format((behavior_policy.argmax(axis=1) == mcp_greedy.argmax(axis=1)).sum() / n_states))


# In[23]:


if plot and UNSAFETY_PROB == 1.0:
    sns.histplot(scipy.stats.entropy(behavior_policy, axis=1))
    plt.title('Behavior policy per-state entropy')
    plt.xlabel('Entropy') 
    plt.show()
    
    sns.histplot(scipy.stats.entropy(mcp_softmax, axis=1))
    plt.title('Softmax policy per-state entropy')
    plt.xlabel('Entropy')
    plt.show()
    
    sns.histplot(behavior_policy[mcp_greedy_mask], log_scale=(False, True))
    plt.xlabel('Action probability greedy policy in behavior policy')
    behavior_policy[mcp_greedy_mask].min(), behavior_policy[mcp_greedy_mask].max()
    plt.show()
    
    evaluation_policy = mcp_greedy

    behavior_policy_ranks = np.flip(behavior_policy.argsort(axis=1), axis=1)
    ep_bp_ranks = []
    for s in range(n_states):
        ep_a = evaluation_policy[s,:].argmax()
        bp_rank = np.where(behavior_policy_ranks[s, :] == ep_a)[0][0]
        ep_bp_ranks.append(bp_rank)

    sns.histplot(ep_bp_ranks, bins=60)
    plt.title('Greedy policy action ranks in behavior policy')
    plt.xlabel('Rank')
    plt.show()

    behavior_policy_ranked_probs = np.flip(np.sort(behavior_policy, axis=1), axis=1)
    ep_bp_prob_mass = []
    for s in range(n_states):
        ep_a = evaluation_policy[s,:].argmax()
        bp_rank = np.where(behavior_policy_ranks[s, :] == ep_a)[0][0]
        ep_bp_prob_mass.append(behavior_policy_ranked_probs[s, 0:bp_rank].sum())

    sns.histplot(ep_bp_prob_mass)
    plt.title('Probability mass up to greedy actduion')
    plt.xlabel('Action probs')
    plt.show()
    np.array(ep_bp_prob_mass).min(), np.array(ep_bp_prob_mass).max()


# In[25]:


joblib.dump(mcp_greedy, "models2/mcp_greedy_policy_{}_{}_{}_{}.bin".format(seed, SHAPING, COMPLIANCE_SCALAR, UNSAFETY_PROB), compress=True)
joblib.dump(mcp_softmax, "models2/mcp_softmax_policy_{}_{}_{}_{}.bin".format(seed, SHAPING, COMPLIANCE_SCALAR, UNSAFETY_PROB), compress=True)


# In[26]:


print("done:", seed, SHAPING, COMPLIANCE_SCALAR, UNSAFETY_PROB)


# In[ ]:




