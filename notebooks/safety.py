import numpy as np
import scipy
import math
import config
import itertools
import utils
from config import fio2_peep_table, fio2_bins, peep_bins, tv_bins


# STATE COMPLIANCE RULES

def rr_compliance_clinical(data):
    """
    Returns True/False to indicate whether the respiratory rate values for a
    given dataset are according to a clinically informed rr compliance rule.
    """
    return data.resprate_imp_scaled_impknn_unscaled <= 35.0

def spo2_compliance_clinical(data):
    """
    Returns True/False to indicate whether the SpO2 vlaues for a given dataset
    are according to a clinically informed spo2 compliance rule.
    """
    return data.spo2_imp_scaled_impknn_unscaled >= 88.0

def pplat_compliance_clinical(data):
    """
    Returns True/False to indicate whether the plateau pressure values for a
    given dataset are according to a clinically informed pplat compliance rule.
    """
    return data.plateau_pressure_imp_scaled_impknn_unscaled < 30

def ph_compliance_clinical(data):
    """
    Returns True/False to indicate whether the ph values for a given dataset
    are according to a clinically informed ph compliance rule.
    """
    return (data.ph_imp_scaled_impknn_unscaled > 7.2) & (data.ph_imp_scaled_impknn_unscaled < 7.5)

def state_compliance_clinical(data, aggregator):
    """
    Function that returns, for a given dataset and a given aggregation
    function, a compliance result for the states in that dataset.  The return
    type of this result depends on the aggregator and can be boolean, float,
    etc.
    """
    compliance_fs = [
        rr_compliance_clinical,
        spo2_compliance_clinical,
        pplat_compliance_clinical,
        ph_compliance_clinical,
    ]
    compliances = [f(data) for f in compliance_fs]
    return aggregator(compliances)


# STATE COMPLIANCE AGGREGATORS
def any_clinical_timestep(compliances):
    """
    Function that aggregates a combination of compliance rule results by an
    inclusive `or` operation.
    """
    return np.any(compliances, axis=0)

def all_clinical_timestep(compliances):
    """
    Function that aggregates a combination of compliance rule results by an
    `and` operation.
    """
    return np.all(compliances, axis=0)

def avg_clinical_timestep(compliances):
    """
    Function that aggregates a combination of compliance rule results by taking
    their average.
    """
    return np.mean(compliances, axis=0)


# ACTION COMPLIANCE RULES
def action_compliance_clinical(data):
    """
    Returns a pandas.Series of booleans indicating whether the actions for a
    given dataset are according to the clinically informed compliance rules.
    """
    return data.action_discrete.apply(lambda x: action_compliance_map[x])

def tv_compl_clinical(tv, fio2, peep):
    """
    Returns True/False to indicate whether the tidal volume values for a given
    dataset are according to a clinically informed tv compliance rule.
    """
    return tv <= 8.5

def peep_compl_clinical(tv, fio2, peep):
    """
    Returns True/False to indicate that the input action, i.e. a combination of
    tv, fio2 and peep settings, is according to the clinically fomred peep
    rules.
    """
    known_fio2 = utils.to_known_fio2(fio2)
    return peep >= fio2_peep_mins[known_fio2] and peep <= fio2_peep_maxs[known_fio2]

def fio2_compl_clinical(tv, fio2, peep):
    """
    Returns True/False to indicate that the input action, i.e. a combination of
    tv, fio2 and peep settings, is according to the clinically informed fio2
    compliance rules.
    """ 
    return fio2 >= config.fio2_min and fio2 <= config.fio2_max

def action_compl_clinical(tv, fio2, peep):
    """
    Returns True/False to indicate that the input action, i.e. a combination of
    tv, fio2 and peep settings, is according to the clinically informed
    compliance rules.
    """
    return fio2_compl_clinical(tv, fio2, peep) and peep_compl_clinical(tv, fio2, peep) and tv_compl_clinical(tv, fio2, peep)


# POLICY SAFETY UTILITIES
def safe_action_policy(policy, safety_map, unsafety_score=0.0):
    """
    Makes a given policy safe according to a given safety action-id -> bool
    safety mapping.

    Returns unnormalized scores where the score for unsafe actions are set to the min 
    has been set to 0.0 and all other probabilities have not been rescaled.
    """
    if unsafety_score is None:
        unsafety_score = policy.min()
    compliant_score = policy.copy()
    for action_id in range(policy.shape[1]):#range(7**3):
        if not safety_map[action_id]:
            compliant_score[:, action_id] = unsafety_score
    return compliant_score

def repaired_safe(policy, default_policy, greedy=False, safety_map=None):
    """
    Makes a given `policy` safe by setting all probabilities of the safety indicator
    array `safety_map` to 0.0 and setting all action probabilities to `default_policy`
    for states with all-zero action probabilities. Note that if `default_policy` is not safe,
    the resulting policy will not be safe.
    """
    repair_func = utils.repair_policy_greedy if greedy else utils.repair_policy
    if safety_map is None:
        safety_map = action_compliance_map
    return utils.normalize_policy_probs(
            repair_func(
                safe_action_policy(policy, safety_map),
                default_policy
                )
            )

def repaired_safe_soft(policy, default_policy, unsafe_max_prob=0.0, greedy=False, safety_map=None,):
    """
    Makes a given `policy` more safe by setting all probabilities of the safety
    indicator array `safety_map` to sum to `unsafe_max_prob` and setting all
    action probabilities to `default_policy` for states with all-zero action
    probabilities. Note that if `default_policy` is not safe, the resulting
    unsafe action probabilities may sum to a probability > unsafe_max_prob.
    """
    repair_func = utils.repair_policy_greedy if greedy else utils.repair_policy
    if safety_map is None:
        safety_map = action_id_compliance
    safety_map = np.array(safety_map)
    safe_policy = np.zeros(policy.shape)
    for state_id in range(policy.shape[0]):
        total_state_unsafe_prob = (policy[state_id,:] * ~safety_map).sum()
        total_state_safe_prob = (policy[state_id,:] * safety_map).sum()
        safe_max_prob = total_state_safe_prob
        unsafe_scalar = min(total_state_unsafe_prob, unsafe_max_prob)
        safe_scalar = 1 - unsafe_scalar
        for action_id, safety in enumerate(safety_map):
            if not safety:
                if total_state_unsafe_prob > 0.0:
                    safe_policy[state_id, action_id] = (policy[state_id, action_id] / total_state_unsafe_prob) * unsafe_scalar
                else:
                    safe_policy[state_id, action_id] = 0.0
            else:
                if total_state_safe_prob > 0.0:
                    safe_policy[state_id, action_id] = (policy[state_id, action_id] / total_state_safe_prob) * safe_scalar
                else:
                    safe_policy[state_id, action_id] = 0.0
    return_policy = repair_func(
            safe_policy,
            default_policy)
    assert (1 - return_policy.sum(axis=1)).min() < 1e-10 , "policy action probs should sum to ~1.0 for all states"
    return return_policy


# SAFETY DEFINITION UTILITIES
fio2_peep_mins = {}
fio2_peep_maxs = {}
for fio2, peep in config.fio2_peep_table:
    if fio2 not in fio2_peep_mins:
        assert fio2 not in fio2_peep_maxs
        fio2_peep_mins[fio2] = peep
        fio2_peep_maxs[fio2] = peep
    else:
        fio2_peep_mins[fio2] = min(fio2_peep_mins[fio2], peep)
        fio2_peep_maxs[fio2] = max(fio2_peep_maxs[fio2], peep)

# the lower bounds for the three variables in the action space
lower_bounds = [[var[0] for var in ranges] for ranges in config.action_bin_definition]

action_compliance_map = {}
# first check whether policy adheres to Table of Paired FiO2 and PEEP settings at https://litfl.com/ardsnet-ventilation-strategy/
action_id_compliance = []
for action_id in range(7**3):
    tv_range, fio2_range, peep_range = utils.to_action_ranges(action_id)
    combos = itertools.product(tv_range, fio2_range, peep_range)
    action_compliance_map[action_id] = any((action_compl_clinical(*c) for c in combos))
    action_id_compliance.append(action_compliance_map[action_id])

action_id_compliance = np.array(action_id_compliance)
