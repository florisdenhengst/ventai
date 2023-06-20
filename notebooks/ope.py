import numpy as np
import pandas as pd
import math
from functools import partial
import itertools

def unscale_result(value, original_range=(-100, 100)):
    return value * (original_range[1] - original_range[0])+original_range[0]

def ois_sa_weights(dataset, e_policy, b_policy):
    # definition ois weights
    #     ois_weights = e_policy / b_policy
    assert not ((b_policy == 0.0) & (e_policy > 0.0)).any(), "Evaluation policy should have some support in behavior policy"
    pi_b_as = dataset.apply(lambda x: b_policy[x.state, x.action_discrete], axis=1)
    pi_e_as = dataset.apply(lambda x: e_policy[x.state, x.action_discrete], axis=1)
    ois_weights = pi_e_as / pi_b_as
    return ois_weights

def ois_traj_weights(dataset, e_policy, b_policy):
    # store ois weight for each state-action pair in the trajectory
    ds = dataset.copy()
    ds['ois_weight'] = ois_sa_weights(ds, e_policy, b_policy)
    traj_weights = ds.groupby('icustay_id')['ois_weight'].transform('prod')
    return traj_weights

def ois_value_trajectory(dataset, e_policy, b_policy):
    """
    Ordinary trajectory-based importance sampling as defined by Sutton & Barto 2nd ed., eq. 5.4 on page 104.
    """
    traj_weights = ois_traj_weights(dataset, e_policy, b_policy)
    # multipliy importance weights in trajectory to determine trajectory importance weight (IW-T)
    # multiply traj IW-T with discounted return of trajectory
    ois_returns = traj_weights * dataset.traj_return
    return ois_returns

def ois_policy(dataset, e_policy, b_policy):
    ds = dataset.copy()
    ds['ois_returns'] = ois_value_trajectory(dataset, e_policy, b_policy)
    ds['ois_traj_weights'] = ois_traj_weights(dataset, e_policy, b_policy)
    ois_returns_stay = ds.groupby('icustay_id').first()['ois_returns']
    traj_weights_stay = ds.groupby('icustay_id').first()['ois_traj_weights']
    expectation = ois_returns_stay.mean()
    variance = np.var( ois_returns_stay * traj_weights_stay) / len(dataset)
    return (expectation, variance)

def phwis_policy(dataset, e_policy, b_policy):
    ds = dataset.copy()
    ds['traj_weights'] = ois_traj_weights(dataset, e_policy, b_policy)
    ds['traj_value'] = ois_value_trajectory(dataset, e_policy, b_policy)
    weights_vals = ds.groupby('icustay_id')[['traj_weights', 'traj_value', 'traj_len']].first()
    # TODO FdH: what is a 'fair' default? 0.0 is according to S&B p. 105, but it does not take into account negative reward    weights_vals[
    if weights_vals['traj_weights'].sum() == 0:
        wis_policy = 0 # according to S&B p 105
    else:
        wis_policy = 0
        n_total_trajs = dataset.icustay_id.nunique()
        traj_lens = ds.groupby('icustay_id').traj_len.first()
        for l in traj_lens.unique():
            subset = weights_vals[weights_vals.traj_len == l]
            if subset['traj_weights'].sum() > 0.0:
                wis_policy += (len(subset)/ n_total_trajs) * (subset['traj_value'].sum() / subset['traj_weights'].sum())
    # determine WIS variance estimation according to "Aslett, Coolen & De Bock", page 30
    var = wis_var(wis_policy, ds.groupby('icustay_id')['traj_return'].first(), weights_vals['traj_weights'])
    return wis_policy, var, ds.groupby('icustay_id')['traj_weights'].first()

def wis_var(wis_policy, traj_returns, traj_weights):
    """
    Inputs:
    * wis policy: OPE estimation
    * traj_returns: returns of trajectories as observed in the dataset
    * traj_weights: OPE estimation weights
    According to "Aslett, Coolen & De Bock", page 30
    """
    traj_w_sq_norm = (traj_weights / traj_weights.sum()) ** 2
    traj_sq_err = (traj_returns - wis_policy) ** 2
    assert traj_w_sq_norm.shape == traj_sq_err.shape, "Inputs should have compatible shape"
    assert traj_w_sq_norm.shape[0] == len(traj_returns), "Inputs should have compatible shape"
    sigm_sq = (traj_w_sq_norm * traj_sq_err).sum()
    variance = sigm_sq.mean()
    return variance

def wis_policy(dataset, e_policy, b_policy):
    """
    Weighted trajectory-based importance sampling as defined by Sutton & Barto, page 104/105.
    """
    ds = dataset.copy()
    ds['traj_weights'] = ois_traj_weights(dataset, e_policy, b_policy)
    ds['traj_value'] = ois_value_trajectory(dataset, e_policy, b_policy)
    weights_vals = ds.groupby('icustay_id')[['traj_weights', 'traj_value']].first()
    # TODO FdH: what is a 'fair' default? 0.0 is according to S&B p. 105, but it does not take into account negative reward    weights_vals[
    if weights_vals['traj_weights'].sum() == 0:
        wis_policy = 0 # according to S&B p 105
    else:
        wis_policy = weights_vals['traj_value'].sum() / weights_vals['traj_weights'].sum()
    # determine WIS variance estimation according to "Aslett, Coolen & De Bock", page 30
    var = wis_var(wis_policy, ds.groupby('icustay_id')['traj_return'].first(), weights_vals['traj_weights'])
    return wis_policy, var, ds.groupby('icustay_id')['traj_weights'].first()


def wdr_inner(ds, e_policy, b_policy, gamma):
    sa_weights = ois_sa_weights(ds, e_policy, b_policy)
    assert sa_weights.isna().sum() == 0, "sa weight is nan somewhere in wdr_inner"
    ds['sa_weights'] = sa_weights / sa_weights.sum()
    first_states = ds.sort_values(['icustay_id', 'start_time']).groupby('icustay_id').first()
    term_a = first_states.v_estimate.mean()
    assert not np.isnan(term_a), "term_a is nan in wdr_inner"
    ds['term_b'] = ds.reward - ds.q_estimate + gamma * ds.next_v_estimate
    assert ds['term_b'].isna().sum() == 0, "term_b is nan somewhere in wdr_inner"
    ds['weighted_sa'] = ds.gamma * ds.sa_weights * ds.term_b
    weighted_traj = ds.groupby('icustay_id').weighted_sa.sum()
    assert not np.isnan(weighted_traj).any(), "weighted_traj is somewhere nan in wdr_inner"
    summed_weighted_traj = weighted_traj.sum()
    assert not np.isnan(summed_weighted_traj), "summed_weighted_traj is nan in wdr_inner"
    return term_a, term_a + summed_weighted_traj

def wdr_policy(dataset, e_policy, b_policy, q_e_estimator, v_e_estimator, gamma):
    """
    Weighted doubly robust estimator as introduced in Equation 2 of ''Data-Efficient
    Off-Policy Policy Evaluation for Reinforcement Learning'' by Thomas & Brunskill 2016.
    Takes as input a dataset of trajectories, an evaluation policy, a behavior policy, a
    state-value estimator for the evaluation policy 'v_e_estimator' and a state-action value
    estimator 'q_e_estimator'.
    """
    ds = dataset.copy()
    ds['v_estimate'] = ds.state.apply(v_e_estimator)
    assert ds['v_estimate'].isna().sum() == 0, "V estimate is nan somehwere"
    ds['next_v_estimate'] = ds.state.apply(v_e_estimator).shift(-1).fillna(0.0) # reward is 0 for terminal absorbing states
    ds['q_estimate'] = ds[['state', 'action_discrete']].apply(q_e_estimator, axis=1)
    assert ds['q_estimate'].isna().sum() == 0, "Q estimate is nan somehwere"
    ds['gamma'] = ds.traj_count.apply(lambda x: gamma ** x)
    return wdr_inner(ds, e_policy, b_policy, gamma)

def phwdr_policy(dataset, e_policy, b_policy, q_e_estimator, v_e_estimator,  gamma):
    """
    Per-horizon weighted doubly robust estimator as introduced in Appendix A of
    ''Behaviour Policy Estimation in Off-Policy Policy Evaluation: Calibration
    Matters'' by Raghu et al. 2018.
    Takes as input a dataset of trajectories, an evaluation policy, a behavior policy, a
    state-value estimator for the evaluation policy 'v_e_estimator' and a state-action value
    estimator 'q_e_estimator'.
    """
    # TODO FdH: implement
    ds = dataset.copy()
    ds['v_estimate'] = ds.state.apply(v_e_estimator)
    ds['next_v_estimate'] = ds.state.apply(v_e_estimator).shift(-1).fillna(0.0) # reward is 0 for terminal absorbing states
    ds['q_estimate'] = ds[['state', 'action_discrete']].apply(q_e_estimator, axis=1)
    ds['gamma'] = ds.traj_count.apply(lambda x: gamma ** x)
    n = ds.icustay_id.nunique()
    phwdr_result = 0.0
    for l in ds.traj_len.unique():
        subset = ds[ds.traj_len == l]
        n_subset = subset.icustay_id.nunique()
        _, wdr_inner_result = wdr_inner(subset, e_policy, b_policy, gamma)
        phwdr_inner =  wdr_inner_result * (n_subset / n)
        assert not np.isnan(phwdr_result), "phwdr is nan for length {}".format(l)
        phwdr_result += phwdr_inner
    return phwdr_result


def infer_estimators_tabular(dataset, policy, gamma, k):
    """
    Infers Q and V estimators for a given policy using FQE.
    """
    n_actions = policy.shape[1]
    n_states = policy.shape[0]
    q_estimate = np.zeros(shape=(n_states, n_actions))
    v_estimate = np.zeros(shape=n_states)
    for i in range(k):
        targets = [[[] for _ in range(n_actions)] for _ in range(n_states)]
        for row in dataset[['state', 'action_discrete', 'reward', 'next_state']].itertuples():
            s, a, r, ns = row.state, row.action_discrete, row.reward, row.next_state
            nqv = 0
            assert not np.isnan(r), "reward is nan for {}".format((s,a,ns))
            if ns < n_states:
                for a in range(n_actions):
                    assert not np.isnan(q_estimate[ns, a]), "q_estimate is nan for {}".format((s,a))
                    assert not np.isnan(policy[ns, a]), "policy is nan for {}".format((s,a))
                    nqv += q_estimate[ns, a] * policy[ns, a]
                targets[s][a].append(r + gamma * nqv)
            else:
                targets[s][a].append(r)
        for s, a in itertools.product(range(n_states), range(n_actions)):
            if len(targets[s][a]) > 0:
                q_estimate[s, a] = sum(targets[s][a]) / len(targets[s][a])
    for s in range(n_states):
        v_estimate_s = 0
        for a in range(n_actions):
            assert np.isnan(policy[s, a]).sum() == 0, "policy is nan for {}".format((s,a))
            assert np.isnan(q_estimate[s, a]) == 0, "q_estimate is nan for {}".format((s,a))
            v_estimate_s += q_estimate[s, a] * policy[s, a]
        v_estimate[s] = v_estimate_s
        assert not np.isnan(v_estimate[s]), "v estimate is nan for {}".format(s)
    return q_estimate, v_estimate

def infer_estimators_func(dataset, policy, gamma, k):
    def get_from_tabular_sa(tabular, sa):
        return tabular[sa[0], sa[1]]
    def get_from_tabular_s(tabular, state):
        return tabular[state]
    q_tabular, v_tabular = infer_estimators_tabular(dataset, policy, gamma, k)
    return partial(get_from_tabular_sa, q_tabular), partial(get_from_tabular_s, v_tabular)


def inner_ess(importance_weights):
    """
    helper function to compute effective sample sizes.
    """
    normalized_importance_weights = importance_weights / importance_weights.sum()
    return (normalized_importance_weights ** 2).sum()

def ess(importance_weights):
    """
    Returns the effective sample size according to the standard definition for importance sampling by Kong (1992).
    """
    ess = inner_ess(importance_weights)
    if ess == 0.0:
        return 0.0
    else:
        return 1.0 / inner_ess(importance_weights)

def hcope(dataset, e_policy, b_policy, c, delta, unscale=True, optimized=True):
    """
    High-confidence off policy evaluation for a given dataset, evaluation
    policy, behavior policy, c and delta. Returns the estimated lower bound of the mean
    for a 1-delta confidence level in the original scale of unscale=True.
    If unscale=False, the result is given on the returns scaled to [0,1].
    If optimized=True, it uses an implementation that requires only a single pass over the data.
    """
    assert c > 0, "c parameter should be > 0, given {}".format(c)
    assert 0 < delta < 1, "delta parameter should be in (0,1), given {}".format(delta)
    ds = dataset.copy()
    # scale the trajectory returns to [0, 1]
    return_min, return_max = ds.traj_return.min(), ds.traj_return.max()
    ds['traj_value'] = ois_value_trajectory(ds, e_policy, b_policy)
    traj_values = ds.groupby('icustay_id')['traj_value'].first()
    n = len(traj_values)
#    if c > traj_values.max():
#        raise ValueError("c of {} bigger than max of {}".format(c, traj_values.max()))
    cs = np.repeat(c, n)
    Y = np.minimum(traj_values, cs)
    if optimized:
        hcope_result = _hcope_singlepass(Y, cs, delta)
    else:
        hcope_result = _hcope_thm1(Y, cs, delta)
    if unscale:
        result = unscale_result(hcope_result, original_range=(return_min, return_max))
    else:
        result = hcope_result
    return result

def _hcope_thm1(Y, cs, delta):
    """
    Implemented according to Thm.1 in Thomas, Teocharous and Ghavamzadeh (2015).
    Assumes cs is an array with constant value.
    """
    n = len(Y)
    empirical_mean = Y.mean()
    second_term = ((1 / cs).sum() ** -1) * ((7*n*np.log(2/delta))/(3*(n-1)))
    third_term_sum = 0
    c = cs[0]
    for i in Y:
        for j in Y:
            third_term_sum += ((i/c) - (j/c))**2
    third_term = ((1/ cs).sum() ** -1) * math.sqrt((np.log(2/delta)/(n-1))*third_term_sum)
    return empirical_mean - second_term - third_term

def _hcope_singlepass(Y, cs, delta):
    """
    Implemented according to Remark 3 in Thomas, Teocharous and Ghavamzadeh (2015).
    Takes a single pass over the Y input.
    Assumes cs is an array with constant value.
    """
    n = len(Y)
    c = cs[0]
    Y2 = Y / cs
    term_1 = (n / c) ** -1
    brack_term_1 = Y2.sum() -  (7*n*math.log(2/delta))/(3*(n-1))
    brack_inner_1 = n*(Y2**2).sum()
    brack_inner_2 = Y2.sum() ** 2
    brack_term_2 = math.sqrt(((2*math.log(2/delta))/(n-1)) *(brack_inner_1 - brack_inner_2))
    return term_1 * (brack_term_1 - brack_term_2)

def hcope_prediction(ds, e_policy, b_policy, n_post, c, delta, unscale=True):
    return_min, return_max = ds.traj_return.min(), ds.traj_return.max()
    ds['traj_value'] = ois_value_trajectory(ds, e_policy, b_policy)
    traj_values = ds.groupby('icustay_id')['traj_value'].first()
    n_pre = len(traj_values)
    cs = np.repeat(c, n_pre)
    Y = np.minimum(traj_values, cs)
    term_sample_mean = Y.sum()
    term2 = (7*c*math.log(2/delta))/(3*(n_post-1))
    term3 = math.log(2/delta)/n_post
    term4 = 2 / (n_pre*(n_pre-1))
    term5 = n_pre * (Y**2).sum() - term_sample_mean**2
    hcope_prediction = term_sample_mean - term2 - math.sqrt(term3*term4*term5)
    if unscale:
        result = unscale_result(hcope_prediction, original_range=(return_min, return_max))
    else:
        result = hcope_prediction
    return result

def am(dataset, e_policy, b_policy, delta, unscale=True):
    """
    Implemented according to Anderson Inequality in Thomas, Teocharous and Ghavamzadeh (2015).
    """
    assert 0 < delta < 1
    ds = dataset.copy()
    return_min, return_max = ds.traj_return.min(), ds.traj_return.max()
    ds['traj_return'] = (ds.traj_return + abs(return_min)) / (return_max + abs(return_min))
    assert ds['traj_return'].min() == 0.0
    assert ds['traj_return'].max() == 1.0
    ds['traj_value'] = ois_value_trajectory(ds, e_policy, b_policy)
    traj_values = ds.groupby('icustay_id')['traj_value'].first()
    n = len(traj_values)
    zs = sorted(traj_values)
    maxz = max(zs)
    sum_r = 0
    for i, z in enumerate(zs[:-1]):
        sum_r += (zs[i+1] - z) * min(1, i/n+math.sqrt((np.log(2/delta))/(2*n)))
    am_result = maxz - sum_r
    if unscale:
        return unscale_result(am_result, original_range=(return_min, return_max))
    else:
        return am_result
