pgf = True


import numpy as np 
import matplotlib as mpl
if pgf:  # Before importing matplotlib.pyplot
    mpl.use('pgf')
import matplotlib.pyplot as plt 
import seaborn as sns
from plot_utils import set_size


def map_interval(results, from_interval, to_interval):
    a, b = from_interval 
    c, d = to_interval
    return [c + (d - c)/(b - a) * (x - a) for x in results]

def scale_results(results):
    extremes = np.min(results), np.max(results)
    return map_interval(results, extremes, [0, 1])


mpl.style.use('seaborn')
sns.set_theme(style='whitegrid')
sns.set_context("paper", font_scale=0.7) 
plt.rcParams.update({
    "font.family": "serif",  
    "text.usetex": True,     
    "pgf.rcfonts": False     
    })

textwidth = 347  # pt.
reward = 'target'
results = np.load(f"test-results/case_study_redoparams_{reward}_0.0001_32_4_real.npy", allow_pickle=True).item()
name = f'case_study_scaled_{reward}'

# Keys of dictionairy per method in results and name for label
axis_mapping = { 
    'rew': 'Reward',
    'sal': 'Salary',
    'dev': 'Deviation',
    'soc': 'SoC deviation',
    'oob': 'Number of out of bounds',
    'ill': 'Number of illegal actions',
    'sem': 'Number of semi-illegal actions',
    'tot': 'Total number of hires/fires',
    'cos': 'Total costs'
}

# Keys of results dictionary and name for label
methods = { 
    'unconstrained': 'PPO',
    'mask': 'Mask',
    'penalty': 'Penalty',
    'nofire': 'No-fire',
    'heuristic': 'LP',
}

metrics = ['rew', 'soc', 'ill', 'cos'] if reward == 'combined' else ['rew', 'dev', 'ill', 'cos']

fig, ax = plt.subplots(2, 2, figsize=set_size(textwidth, subplots=(2,2)))
ax = ax.flatten()
for j, typ in enumerate(metrics):
    result = [results[res][typ] for res in results if res in methods]
    result = scale_results(result)
    sns.boxplot(data=result, palette='cubehelix', ax=ax[j], fliersize=0.7)
    ax[j].set_ylabel(axis_mapping[typ])
    ax[j].set_xlabel('Solution method')
    ax[j].set_xticklabels([methods[x] for x in methods])
    ax[j].set_ylim(0,1)
fig.tight_layout()
if pgf:
    plt.savefig(f'Data/Figures/{name}.pgf', format='pgf')
else:
    plt.show()
