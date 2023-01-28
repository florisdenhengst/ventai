import numpy as np
import scipy
import math
import config
import itertools
import config
from config import fio2_peep_table, fio2_bins, peep_bins, tv_bins

def var_to_sem(var, n):
    std = var_to_std(var)
    return std / math.sqrt(n)

def var_to_sem_range(var, mean, n):
    sem = var_to_sem(var, n)
    return mean - sem, mean + sem

def var_to_ci_normal(var, mean, n):
    sem = var_to_sem(var, n)
    return mean - 1.96*sem, mean + 1.96*sem

def var_to_ci_cheb(var, mean, n, k=4.4721):
    std = var_to_std(var)
    return mean - k*std, mean + k*std

def var_to_stddev_range(var, mean):
    std = var_to_std(var)
    return mean - std, mean + std

def var_to_std(var):
    """
    Returns standard normal error for a given variance. Assumes a normal
    distribution.
    """
    return np.sqrt(var)

def mean_ci(mu, var, n, conf=.95):
    """
    Returns a confidence interval for a given mean (mu), variance, number of
    observations (n) and confidence interval.
    Assumes a normal distribution.
    """
    sigma = var_to_std(var) # variance to standard error
    return scipy.stats.norm.interval(conf, loc=mu, scale=sigma)


def to_known_fio2(fio2):
    """
    Returns, for a given fio2 level, the known fio2 level in the ARDS peep-fio2
    table.
    """
    return math.floor(fio2 / 10) * 10

def to_discrete_action(tidal_volume, fio2, peep, action_bin_definition=config.action_bin_definition):
    """
    Returns the action identifier for a particular combination of tv, fio2 and
    peep based on the bins defined in `action_bin_definition`.

    Assumes: * that the order of these variables in the definition is: tv_bin,
    fio2_bin, peep_bin, * that the ranges in the definition are defined as
    [lower bound, upper bound)

    TODO: speed up computation using hash lookups for ranges and vectorization
    with pandas?
    """
    if peep < 0 and peep > -1e-5:
        peep = 0.0
    for i, (tv_range, fio2_range, peep_range) in enumerate(action_bin_definition):
        # extract lower and upper bounds of all ranges
        tv_lb, fio2_lb, peep_lb = tv_range[0], fio2_range[0], peep_range[0]
        tv_ub, fio2_ub, peep_ub = tv_range[1], fio2_range[1], peep_range[1]
        if (tv_lb   <= tidal_volume < tv_ub and
            fio2_lb <= fio2         < fio2_ub and
            peep_lb <= peep         < peep_ub):
            # in range
            return i
    raise ValueError("Action (tv: {}, fio2:{}, peep:{}) not in action space".format(tidal_volume, fio2, peep))

def to_discrete_action_bins(action_id, action_bin_definition=config.action_bin_definition):
    """
    Returns, for a given integer action_id, the corresponding bin indices for
    tv, fio2 and peep.
    """
    tv_range, fio2_range, peep_range = action_bin_definition[action_id]
    tv_bin = tv_bins.index(tv_range)
    fio2_bin = fio2_bins.index(fio2_range)
    peep_bin = peep_bins.index(peep_range)
    return tv_bin, fio2_bin, peep_bin

def to_action_ranges(action_id):
    """
    Returns, for a given action_id, the corresponding (min, max) tuples for tv,
    fio2 and peep settings.
    """
    tv_bin, fio2_bin, peep_bin = to_discrete_action_bins(action_id)
    return tv_bins[tv_bin], fio2_bins[fio2_bin], peep_bins[peep_bin]

def repair_policy(policy, default_policy):
    """
    Returns a copy of `policy` but defaults to `default_policy` for states with
    all-zero action probabilities.
    """
    repaired_policy = policy.copy()
    all_zero_states = np.where((policy.sum(axis=1) == 0.0)| (np.isnan(policy.sum(axis=1))))[0]
    default_policy_set = default_policy.sum(axis=1) > 0.0
    for s in all_zero_states:
        if default_policy_set[s]:
            repaired_policy[s, :] = default_policy[s, :]
        else:
            raise ValueError('Behavior policy all zero for state {}'.format(s))
    return repaired_policy

def repair_policy_greedy(policy, default_policy):
    """
    Returns a copy of evaluation policy which defaults to the best action
    according to behavior policy if evaluation policy has states with no actions.
    """
    repaired_policy = policy.copy()
    all_zero_states = np.where((policy.sum(axis=1) == 0.0)| (np.isnan(policy.sum(axis=1))))[0]
    default_policy_set = default_policy.sum(axis=1) > 0.0
    for s in all_zero_states:
        if default_policy_set[s]:
            repaired_policy[s, :] = 0.0
            greedy_a = np.argmax(default_policy[s,:])
            repaired_policy[s, greedy_a] = 1.0
        else:
            raise ValueError('Behavior policy all zero for state {}'.format(s))
    return repaired_policy

def repair_policy_uniform(policy):
    """
    Return a copy of `policy` in which states with all-zero action probabilities have
    uniform action probabilities instead.
    """
    repaired_policy = policy.copy()
    all_zero_states = np.where((policy.sum(axis=1) == 0.0)| (np.isnan(policy.sum(axis=1))))[0]
    for s in all_zero_states:
        repaired_policy[s,:] = 1 / len(repaired_policy[s,:])
    return repaired_policy

def repair_unsupported_greedy_policy(policy, train_set):
    """
    Resets actions in input policy to the observed action in train_set for all states
    that were visited once in train_set.
    """
    repaired_policy = policy.copy()
    state_visit_counts = train_set.state.value_counts()
    single_visit_states = state_visit_counts[state_visit_counts == 1].index
    for state in single_visit_states:
        observed_action = train_set[train_set.state == state].action_discrete.to_numpy()[0]
        if np.argmax(policy[state, :]) != observed_action:
            repaired_policy[state, :] = 0.0
            repaired_policy[state, observed_action] = 1.0
    return repaired_policy

def normalize_policy_probs(policy):
    """
    Normalizes policy probabilities to sum to 1 for all states.
    """
    assert policy.min() >= 0.0, "Cannot normalize policy with input action probs < 0.0"
    assert policy.max() <= 1.0, "Cannot normalize policy with input action probs > 1.0"
    assert (policy.sum(axis=1) > 0.0).all(), "Cannot normalize policy with all-zero action probability state"
    return policy / np.repeat(policy.sum(axis=1), config.n_actions).reshape(config.n_states, config.n_actions)
