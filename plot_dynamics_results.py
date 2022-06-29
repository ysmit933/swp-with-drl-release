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
sns.set_context("paper", font_scale=0.7, rc={"lines.linewidth": 1})
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })
textwidth = 347  # pt.

mapping = [x / 100 for x in range(11)]
x_axis_mapping = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]

results_target = np.load(f'test-results/dynamics_redo_target_wandb_ppo.npy', allow_pickle=True).item()
results_target_heur = np.load(f'test-results/dynamics_redo_target_wandb_heur.npy', allow_pickle=True).item()
results_combined = np.load(f'test-results/dynamics_redo_combined_wandb_ppo.npy', allow_pickle=True).item()
results_combined_heur = np.load(f'test-results/dynamics_redo_combined_wandb_heur.npy', allow_pickle=True).item()

x_axis = [x for x in range(len(mapping))]

def generate_plot_data(results):
    for x in results:
        types_result = [t for t in results[x]]
        break
    to_plot = {}
    for type_result in types_result:
        to_plot[type_result] = {}
        to_plot[type_result]['mean'] = [np.mean(results[x][type_result]) for x in results]
        std = [np.std(results[x][type_result]) for x in results]
        to_plot[type_result]['std_u'] = [x + y for x, y in zip(to_plot[type_result]['mean'], std)]
        to_plot[type_result]['std_l'] = [x - y for x, y in zip(to_plot[type_result]['mean'], std)]
    return to_plot

target_plot = generate_plot_data(results_target)
target_heur_plot = generate_plot_data(results_target_heur)
combined_plot = generate_plot_data(results_combined)
combined_heur_plot = generate_plot_data(results_combined_heur)

fig, ax = plt.subplots(1, 2, figsize=set_size(textwidth, subplots=(1, 2)))
ax = ax.flatten()

for i, (plot_ppo, plot_heur) in enumerate([(target_plot, target_heur_plot), (combined_plot, combined_heur_plot)]):
    ax[i].fill_between(x_axis, plot_ppo['rew']['std_u'], plot_ppo['rew']['std_l'], alpha=0.5)#, color=colors[i])
    ax[i].plot(plot_ppo['rew']['mean'], label='PPO')#, color=colors[i])
    ax[i].fill_between(x_axis, plot_heur['rew']['std_u'], plot_heur['rew']['std_l'], alpha=0.5)#, color=colors[i])
    ax[i].plot(plot_heur['rew']['mean'], label='Heuristic')#, color=colors[i])
    ax[i].set_ylabel('Reward')
    ax[i].set_xlabel('Relative cohort size')
    ax[i].set_xlabel('Attrition rate')
    ax[i].set_xticks([x*2 for x in range(len(x_axis_mapping))])
    ax[i].set_xticklabels(x_axis_mapping)
    ax[i].legend()
    reward = 'Target' if i == 0 else 'Combined'
    ax[i].set_title(f'{reward} reward')
fig.tight_layout()

if pgf:
    fig.savefig(f'Data/Figures/dynamics_bothrewards.pgf', format='pgf')
else:
    plt.show()
