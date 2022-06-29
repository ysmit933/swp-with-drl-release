import numpy as np
from scipy.stats import ttest_ind


results_target = np.load(f"test-results/test_setting_target.npy", allow_pickle=True).item()
results_combined = np.load(f"test-results/test_setting_combined.npy", allow_pickle=True).item()
ppo_target_rewards = results_target['ppo']['rew']
heur_target_rewards = results_target['heuristic']['rew']
ppo_combined_rewards = results_combined['ppo']['rew']
heur_combined_rewards = results_combined['heuristic']['rew']

# H_0: mean rewards PPO = mean rewards Heuristic, H_a: mean rewards PPO < mean rewards Heuristic
test_target = ttest_ind(ppo_target_rewards, heur_target_rewards, alternative='less')
# H_0: mean rewards PPO = mean rewards Heuristic, H_a: mean rewards PPO > mean rewards Heuristic
test_combined = ttest_ind(ppo_combined_rewards, heur_combined_rewards, alternative='greater')

print("Target reward p-value: ", test_target[1])  # 7.055577844002918e-38
print("Combined reward p-value: ", test_combined[1])  # 1.3386611688941234e-48
