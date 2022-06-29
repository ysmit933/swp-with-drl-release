pgf = False


import numpy as np 
import matplotlib as mpl
if pgf:  # Before importing matplotlib.pyplot
    mpl.use('pgf')
import matplotlib.pyplot as plt 
import seaborn as sns
from plot_utils import set_size


mpl.style.use('seaborn')
sns.set_theme(style='darkgrid')
sns.set_context("paper", font_scale=0.6, rc={"lines.linewidth": 1})
plt.rcParams.update({
    "font.family": "serif",  
    "text.usetex": True,    
    "pgf.rcfonts": False     
    })
textwidth = 347  # pt.

reward = 'target'
results = np.load(f"test-results/test_setting_{reward}.npy", allow_pickle=True).item()
name = f'test_setting_{reward}'

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
    'ppo': 'PPO',
    'heuristic': 'Heuristic',
}

metrics = ['rew', 'soc', 'ill', 'cos'] if reward == 'combined' else ['rew', 'dev', 'ill', 'cos']

fig, ax = plt.subplots(2, 2, figsize=set_size(textwidth, subplots=(2,2)))
ax = ax.flatten()
for j, typ in enumerate(metrics):
    result = [results[res][typ] for res in results if res in methods]
    sns.boxplot(data=result, palette='cubehelix', ax=ax[j], fliersize=0.7)
    ax[j].set_ylabel(axis_mapping[typ])
    ax[j].set_xlabel('Solution method')
    ax[j].set_xticklabels([methods[x] for x in methods])
    # ax[j].legend()
fig.tight_layout()
if pgf:
    plt.savefig(f'Data/Figures/{name}.pgf', format='pgf')
else:
    plt.show()
