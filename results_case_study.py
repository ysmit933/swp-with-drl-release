import numpy as np
from scipy import stats
from scipy.stats import ttest_ind


metric = 'rew'

len_episode = 60
n_cohorts = 10
n_decisions = len_episode * n_cohorts

def scale_result(result, from_range, to_range):
    a, b = from_range 
    c, d = to_range
    return [c + (d - c)/(b-a) * (x- a) for x in result]

# the max upper limits for varying mobility
def scale_target(value):
    # lb = min([min(results_target[variant][metric]) for variant in results_target.keys()])
    lb = 0  # Use 0 for target reward
    ub = max([max(results_target[variant][metric]) for variant in results_target.keys()])
    return scale_result(value, [lb, ub], [0, 1])

def scale_combined(value):
    lb = min([min(results_combined[variant][metric]) for variant in results_combined.keys()])
    ub = max([max(results_combined[variant][metric]) for variant in results_combined.keys()])
    return scale_result(value, [lb, ub], [0, 1])

def scale_case_target(value):
    # lb = min([min(usecase_target[variant][metric]) for variant in usecase_target.keys()])
    lb = 0  # Use 0 for target reward
    ub = max([max(usecase_target[variant][metric]) for variant in usecase_target.keys()])
    return scale_result(value, [lb, ub], [0, 1]) 

def scale_case_combined(value):
    lb = min([min(usecase_combined[variant][metric]) for variant in usecase_combined.keys()])
    ub = max([max(usecase_combined[variant][metric]) for variant in usecase_combined.keys()])
    return scale_result(value, [lb, ub], [0, 1])

def scale_illegal_actions(value):
    return value / n_decisions * 100.0

def no_scale(value):
    return value

def get_mean_ci(result_set, scale, interval=False, ci_level=0.95):
    """
    Returns mean +- confidence interval at ci_level>
    """
    result_set = scale(np.array(result_set)) 
    mu, sigma = np.mean(result_set), np.std(result_set)
    if interval:
        ci = stats.norm.interval(ci_level, loc=mu, scale=sigma)
    else:
        ci = sigma * 2
    return mu, ci


results_target = np.load(f"test-results/test_setting_target.npy", allow_pickle=True).item()
results_combined = np.load(f"test-results/test_setting_combined.npy", allow_pickle=True).item()

usecase_target = np.load(f"test-results/case_study_redoparams_target_0.0001_32_4_real.npy", allow_pickle=True).item()
usecase_combined = np.load(f"test-results/case_study_redoparams_combined_0.0001_32_4_real.npy", allow_pickle=True).item()

ppo_target_rewards = results_target['ppo'][metric]
heur_target_rewards = results_target['heuristic'][metric]
ppo_combined_rewards = results_combined['ppo'][metric]
heur_combined_rewards = results_combined['heuristic'][metric]

result_sets = [
        ('ppo_t', ppo_target_rewards, scale_target),
        ('heur_t', heur_target_rewards, scale_target),
        ('ppo_c', ppo_combined_rewards, scale_combined),
        ('heur_c', heur_combined_rewards, scale_combined),
        ]

for variant in usecase_target.keys():
    # hack to only include heuristic and unconstrained
    if variant in {'heuristic', 'unconstrained', 'mask', 'penalty', 'nofire'}:
        scale_by = scale_illegal_actions if metric == 'ill' else scale_case_target
        result_sets += [
                ('case_t_' + variant, usecase_target[variant][metric], scale_by),
                ]

for variant in usecase_combined.keys():
    if variant in {'heuristic', 'unconstrained', 'mask', 'penalty', 'nofire'}:
        scale_by = scale_illegal_actions if metric == 'ill' else scale_case_combined
        result_sets += [
                ('case_c_' + variant, usecase_combined[variant][metric], scale_by),
                ]

for name, result_set, scale_by in result_sets:
    mean, ci = get_mean_ci(result_set, interval=False, scale=scale_by)
    print('{}: {:.2f} \pm {:.3f}'.format(name, mean, ci))

