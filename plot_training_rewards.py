pgf = False


import numpy as np 
import matplotlib as mpl
if pgf:  # Before importing matplotlib.pyplot
    mpl.use('pgf')
import matplotlib.pyplot as plt 
import seaborn as sns
from plot_utils import set_size
from stable_baselines3.common.results_plotter import load_results, ts2xy


mpl.style.use('seaborn')
sns.set_theme(style='whitegrid')
sns.set_context("paper")#, font_scale=0.8, rc={"lines.linewidth": 0.7}) 
plt.rcParams.update({
    "font.family": "serif",  
    "text.usetex": True,     
    "pgf.rcfonts": False     
    })

textwidth = 347  # pt.
short = True

models = [
    f'test_setting_target',
    f'test_setting_combined',
    f'case_study_exp_target_unconstrained_1000000_redo_32_4_0.0001',
    f'case_study_exp_combined_unconstrained_1000000_redo_32_4_0.0001',
]

labels = ['Simulated - Operational', 'Simulated - Strategic', 'Real-life - Operational', 'Real-life - Strategic']

def rolling_average(arr, steps=10):
    return [np.mean(arr[max(0, i-steps+1):i+1]) for i in range(len(arr))]

def scale_results(results, from_range, to_range):
    a, b = from_range 
    c, d = to_range
    return [c + (d - c)/(b-a) * (x - a) for x in results]

fig, ax = plt.subplots(1, 1, figsize=set_size(textwidth, subplots=(1, 1)))
# ax = ax.flatten()    

min_target, max_target = 1, 0
for i, model in enumerate([models[0], models[2]]):  # target models
    log_dir = f'./monitor/PPO/{model}'
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    if i == 1:
        x = x[:len(x)//2]
        y = y[:len(y)//2]
    minimum, maximum = np.min(y), np.max(y)
    if maximum > max_target:
        max_target = maximum
    if minimum < min_target:
        min_target = minimum

min_combined, max_combined = 2, -1000
for i, model in enumerate([models[1], models[3]]):
    log_dir = f'./monitor/PPO/{model}'
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    if i == 1:
        x = x[:len(x)//2]
        y = y[:len(y)//2]
    minimum, maximum = np.min(y), np.max(y)
    if maximum > max_combined:
        max_combined = maximum
    if minimum < min_combined:
        min_combined = minimum

for i, (model, name) in enumerate(zip(models, labels)):
    log_dir = f'./monitor/PPO/{model}'
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    if i > 1:
        x = x[:len(x)//2]
        y = y[:len(y)//2]
    if i in [0, 2]:
        y = scale_results(y, [min_target, max_target], [0, 1])
    else:
        y = scale_results(y, [min_combined, max_combined], [0, 1])
    y = rolling_average(y, steps=50)
    ax.plot(x, y, label=name)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Cumulative Reward')
ax.legend(loc=4)
ax.set_title('Training rewards')

fig.tight_layout()
if pgf:
    fig.savefig(f'Data/Figures/training_rewards.pgf', format='pgf')
else:
    plt.show()
