pgf = True


import numpy as np 
import matplotlib as mpl
if pgf:  # Before importing matplotlib.pyplot
    mpl.use('pgf')
import matplotlib.pyplot as plt 
import seaborn as sns
from plot_utils import set_size

mpl.style.use('seaborn')
sns.set_theme(style='whitegrid')
sns.set_context("paper", font_scale=0.7, rc={"lines.linewidth": 0.7}) 
plt.rcParams.update({
    "font.family": "serif",  
    "text.usetex": True,     
    "pgf.rcfonts": False     
    })

textwidth = 347  # pt.

labels = ['LP', 'Unconstrained', 'Mask', 'Penalty', 'No control']
markers = ['o', '^', 's', 'd', 'P']

results_target_rew = [0.99, 0.92, 0.74, 0.87, 0.79] 
results_target_ill = [0.802, 0.330, 0.000, 0.016, 0.000]
results_combin_rew = [0.12, 0.98, 0.75, 0.74, 0.94]
results_combin_ill = [0.491, 0.771, 0.000, 0.000, 0.000]

fig, ax = plt.subplots(1, 2, sharey=True, figsize=set_size(textwidth, subplots=(1, 2)))
for i, (x, y) in enumerate([(results_target_rew, results_target_ill), (results_combin_rew, results_combin_ill)]):
    for j, marker in enumerate(markers):
        ax[i].scatter(x[j], y[j], color='b', marker=marker, label=labels[j])
    ax[i].set_xlabel('Reward')
    if i == 0:
        ax[i].set_ylabel('Violations')
        ax[i].legend()
    # ax[i].set_xlim([0, 1])
    # ax[i].set_ylim([0, 1])
    goal = 'Operational' if i == 1 else 'Strategic'
    ax[i].set_title(goal + ' goal')
    
fig.tight_layout()
if pgf:
    plt.savefig(f'Data/Figures/scatter.pgf', format='pgf')
else:
    plt.show()
