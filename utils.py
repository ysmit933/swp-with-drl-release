"""
Functions used to train models, test models, visualize trainings and the like.
"""
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as th
import torch.multiprocessing as mp
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy

from constrained_policy import ConstrainedACPolicy
from helpers import rolling_average


def load_model(model_name):
    return PPO.load(f'./logs/PPO/{model_name}/best_model.zip')


def test_model(env, model=None, actions=None, state=None, random=False, nothing=False, 
               show=True, random_start=False, pgf=False, pgf_name=None):

    # To allow for environments to be passed in with and without parentheses
    try:
        N = env.n_cohorts
    except AttributeError:
        env = env()
        N = env.n_cohorts

    no_reset = False
    if state is None:
        if random_start:
            obs = env.reset()
            no_reset = True
        else:
            state = env.starting_state
    if not no_reset:
        obs = env.reset(state=state)
    
    goal = env.goal_state

    
    if env.hiring_costs is None:
        env.hiring_costs = [2000] * N
    if env.firing_costs is None: 
        env.firing_costs = [2*x for x in env.hiring_costs]

    if env.reward.specs["method"] == "combined":
        lb = env.reward.specs["lb"]
        ub = env.reward.specs["ub"]
        soc_goal = env.reward.specs["soc"]
    else:
        soc_goal = 7

    N_emps = sum(env.headcounts())
    
    # Metrics to keep track of
    state_history = [env.headcounts()]
    rewards_history = []
    soc_history = []
    hire_history = []
    previous_action = [0] * N
    rewards = 0
    salary_costs = [0] * N
    action_sureness = [0] * N
    total_hires = [0] * N
    total_fires = [0] * N
    number_of_illegal_actions = [0] * N  
    semi_illegal_actions = [0] * N
    deviations = [0] * N
    soc_deviations = 0
    out_of_bounds = 0
    costs = [0] * N

    for t in range(env.time_horizon):
        if actions is not None:
            current_action = actions[t]
            action = [env.hire_options[cohort].index(current_action[cohort]) for cohort in range(N)]
        elif random:
            action = env.action_space.sample()
        elif nothing:
            # Check if firing is possible
            if env.hire_options[0][0] < 0:
                # Assuming same amount of possible hires as possible fires
                action = [len(x) // 2 for x in env.hire_options]
            else:
                action = np.zeros(env.n_cohorts, dtype=int)
        # Action chosen by DRL agent
        else:
            action, _ = model.predict(obs, deterministic=True) 

        # Translate action choice to actual headcounts
        if actions is None:
            current_action = env._action_to_headcount(action)

        # Determine whether action was 'illegal'
        if t >= 1:
            for i in range(N):
                if (previous_action[i] < 0 and current_action[i] > 0) or (previous_action[i] > 0 and current_action[i] < 0):
                    number_of_illegal_actions[i] += 1
        
        # Determine whether action was 'semi-illegal'
        if len(env.state) > N:  # Only if constrained firing
            for i in range(N):
                if (env.state[N+i] == 0 and current_action[i] > 0) or \
                        (env.state[N+i] == 1 and current_action[i] < 0):
                    semi_illegal_actions[i] += 1
        
        # Update metrics based on chosen action
        if hasattr(model, 'policy'):
            current_sureness = model.policy.action_dist.action_quality()
            for i in range(N):
                # TODO: debug
                action_sureness[i] += current_sureness[i]
                # pass

        current_salary_costs = env.current_salary_costs(total=False)
        for i in range(N):
            if current_action[i] < 0:
                total_fires[i] += current_action[i]
            else:
                total_hires[i] += current_action[i]
            salary_costs[i] += current_salary_costs[i]
            costs[i] += total_hires[i] * env.hiring_costs[i] - total_fires[i] * env.firing_costs[i]
        
        # Keep track of current action to determine legality of the next action
        previous_action = current_action

        if len(hire_history) == 0:
            hire_history = np.array(env._action_to_headcount(action))
        else:
            hire_history = np.vstack([hire_history, env._action_to_headcount(action)])

        # Advance the environment one step with the chosen action
        obs, reward, done, _ = env.step(action=action)
        
        # Update variables
        state_history = np.vstack([state_history, env.headcounts()])
        rewards_history.append(reward)
        soc = env.calculate_soc()
        soc_history.append(soc)
        soc_deviations += abs(soc - soc_goal)

        headcounts = env.headcounts()
        for i in range(N):
            deviations[i] += abs(headcounts[i] - goal[i]) / N_emps

        # Number of out of bounds movements
        if env.reward.specs["method"] == "combined":
            for i in range(N):
                if env._state_to_headcount(obs)[i] < lb[i] or env._state_to_headcount(obs)[i] > ub[i]:
                    out_of_bounds += 1

        rewards += reward

        if done:
            break
    
    # Plot results
    if show:
        import matplotlib as mpl
        if pgf:
            mpl.use('pgf')
        import matplotlib.pyplot as plt
        import seaborn as sns

        from plot_utils import set_size

        mpl.style.use('seaborn')
        sns.set_theme(style='darkgrid')
        sns.set_context("paper", font_scale=0.5, rc={"lines.linewidth": 0.5})
        plt.rcParams.update({
            "font.family": "serif",  # use serif/main font for text elements
            "text.usetex": True,     # use inline math for ticks
            "pgf.rcfonts": False,    # don't setup fonts from rc parameters
            "xtick.major.pad": 1,
            "ytick.major.pad": 1,
        })

        states = state_history
        hires = hire_history
        soc_history = np.array(soc_history)
        num_rows = 2
        textwidth = 418  # pt. TODO: redo for paper
        if pgf:
            fig, ax = plt.subplots(num_rows, 3, figsize=set_size(textwidth, subplots=(num_rows, 3)))
        else:
            fig, ax = plt.subplots(num_rows, 3, figsize=(8, 4))
        ax = ax.flatten()

        for i in range(N // 2):  
            ax[0].plot(states[:,i], label=f'M{i+1}')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Headcount (Managers)')
        ax[0].legend()

        for i in range(N // 2):
            ax[1].set_xlabel('Time')
            ax[1].set_ylabel('Number of hires')
            ax[1].plot([h for h in hires[:,i]], label=f'M{i+1}')
            ax[1].legend(loc=1)

        ax[2].set_xlabel('Time')
        ax[2].set_ylabel('Span of control')
        ax[2].plot(soc_history, label='SoC')
        ax[2].legend()

        for i in range(N // 2, N):
            ax[3].set_xlabel('Time')
            ax[3].set_ylabel('Headcount (Contributors)')
            ax[3].plot(states[:,i], label=f'C{i-N//2+1}')
        ax[3].legend(loc=1)

        for i in range(N // 2, N):
            ax[4].set_xlabel('Time')
            ax[4].set_ylabel('Number of hires')
            ax[4].plot([h for h in hires[:,i]], label=f'C{i-N//2+1}')
            ax[4].legend(loc=1)

        if env.reward.specs["method"] == "combined":
            ax[5].set_ylim(-1.05, 2.05)
            reward_label = 'Combined reward'
        else:
            ax[5].set_ylim(-0.05, 1.05)
            reward_label = 'Target reward'

        ax[5].plot(rewards_history, label=reward_label)
        ax[5].set_xlabel('Time')
        ax[5].set_ylabel('Reward')
        ax[5].legend(loc=1)

        fig.tight_layout()
        if pgf:
            if not pgf_name:
                pgf_name = 'default-pgf'
            fig.savefig(f'Data/Figures/{pgf_name}.pgf', format='pgf')
        else:
            plt.show()

    if hasattr(model, 'policy'):
        action_sureness = [x.mean().item() / env.time_horizon for x in action_sureness]

    return {
        'average_reward': rewards / env.time_horizon,
        'salary_costs': salary_costs,
        'total_hires': total_hires,
        'total_fires': total_fires,
        'illegal_actions': number_of_illegal_actions,
        'semi_illegal_actions': semi_illegal_actions,
        'action_sureness': action_sureness,
        'deviations': deviations,
        'soc_deviations': soc_deviations,
        'out_of_bounds': out_of_bounds,
        'mean_state': np.mean(state_history, axis=0),
        'mean_soc': np.mean(soc_history),
        'total_costs': costs
    }
    

def plot_training_rewards(model_name, roll_step=50, show=True, save=True):
    """Plot the training rewards saved by the Monitor wrapper.

    :param log_dir: Path to the directory containing the monitor.csv files
    :param roll_step: Step size to be used for the rolling average. Defaults to 50.
    :param show: If set to True, shows the plotted figure.
    :param save: If set to True, saved the plotted figure into 
        Data/Figures/agent_model_training_rewards.png where 'agent' is the RL 
        agent used for training and 'model' is the name of trained model
    """
    log_dir = f'./monitor/PPO/{model_name}'
    agent_name = 'PPO'
    # _, _, agent_name, model_name = log_dir.split('/')
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    y = rolling_average(y, steps=roll_step)
    plt.plot(x, y, label=agent_name)
    plt.xlabel('Timestep')
    plt.ylabel('Training reward')
    plt.legend()
    if save:    
        if not os.path.exists(f'Data/Figures'):
            os.makedirs(f'Data/Figures')
        plt.savefig(f'Data/Figures/{agent_name}_{model_name}_training_rewards.png')
    if show:
        plt.show()


def plot_multiple_training_rewards(models, names, half=False):
    colors = list(sns.color_palette('deep')) * len(models)
    for i, (model, name) in enumerate(zip(models, names)):
        log_dir = f'./monitor/PPO/{model}'
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if half:
            x = x[:len(x)//2]
            y = y[:len(y)//2]
        y = rolling_average(y, steps=50)
        plt.plot(x, y, label=name, color=colors[i])
        plt.xlabel('Timestep')
        plt.ylabel('Training reward')
    plt.legend()
    plt.show()


def plot_evaluation_rewards(model_name, show=True, save=True):
    """Plot the evaluation rewards obtained by the callback function during training.

    :param log_dir: Path to the directory containing the evaluations.npz file (and best_model.zip)
    :param show: If set to True, shows the plotted figure
    :param save: If set to True, saves the plotted figure into 
        Data/Figures/agent_model_evaluation_rewards.png where 'agent' is the RL 
        agent used for training and 'model' is the name of trained model
    """
    log_dir = f'./logs/PPO/{model_name}'
    _, _, agent_name, model_name = log_dir.split('/')
    data = np.load(f'{log_dir}/evaluations.npz', allow_pickle=True)
    x = data['timesteps']
    y = [tests.mean() for tests in data['results']]
    plt.plot(x, y, label=agent_name)
    plt.xlabel('Timestep')
    plt.ylabel('Evaluation reward')
    plt.legend()
    if save:    
        if not os.path.exists(f'Data/Figures'):
            os.makedirs(f'Data/Figures')
        plt.savefig(f'Data/Figures/{agent_name}_{model_name}_evaluation_rewards.png')
    if show:
        plt.show()


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def train_model(env, time_steps=1e6, log_interval=10, eval_freq=100, eval_env_time_horizon=32,
                policy_kwargs=None, agent_kwargs=None, env_kwargs=None, model_name=None, 
                constrain_firing=False, show=False, use_wandb_entity=None, project_name=None):
    """
    Train a model on the provided environment. During training, the training
    rewards are stored in './monitor/{agent}/{model_name}' and the evaluation
    rewards and the best model are stored in './logs/{agent}/{model_name}.
    """

    # Default network architecture
    if policy_kwargs is None:
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(vf=[256, 256], pi=[128, 128])])

    # Default PPO parameters
    if agent_kwargs is None:
        agent_kwargs = dict(
            learning_rate = linear_schedule(5e-4),
            n_steps = 32,
            batch_size = 4,
            n_epochs = 4,
            gamma = 0.9,
            gae_lambda = 0.9,
            clip_range = 0.2,
            ent_coef = 0.01,
            vf_coef = 0.5,
            policy_kwargs = policy_kwargs,
            verbose = 1,
        )  

    # Default environment parameters
    if env_kwargs is None:
        env_kwargs = dict(
            random_start_percentage = 0.5,
            random_start_around_fixed = True,
            time_horizon=50,
            early_termination=False,
        )
    
    model_name = model_name or 'default-model-name'
    monitor_log_dir = f'./monitor/PPO/{model_name}'
    log_path_dir = f'./logs/PPO/{model_name}'

    n_cpu = mp.cpu_count()

    eval_env_kwargs = copy.deepcopy(env_kwargs)
    eval_env_kwargs['time_horizon'] = eval_env_time_horizon
    eval_env_kwargs['random_start_percentage'] = 0
    eval_freq = agent_kwargs['n_steps'] * n_cpu
    eval_env = env(**eval_env_kwargs)
    eval_env = Monitor(eval_env)
    callback = EvalCallback(eval_env=eval_env, best_model_save_path=log_path_dir, n_eval_episodes=10,
                                 log_path=log_path_dir, eval_freq=eval_freq,
                                 deterministic=True, render=False)
    
    if use_wandb_entity is not None:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        from additional_callback import TensorboardCallback 

        project_name = project_name or 'default-project-name'

        run = wandb.init(
            project=project_name,
            config=agent_kwargs,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True, 
            entity=use_wandb_entity,
            reinit=True,
            name=model_name,
        )
        wandb_callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=1,
        )
        callback = [callback, wandb_callback, TensorboardCallback()]
    
    env = make_vec_env(env, n_envs=n_cpu, env_kwargs=env_kwargs, monitor_dir=monitor_log_dir)
    policy = ConstrainedACPolicy if constrain_firing else 'MlpPolicy'
    
    model = PPO(policy, env, tensorboard_log=f"runs/{run.id}", **agent_kwargs) if use_wandb_entity else PPO(policy, env, **agent_kwargs)
    model.learn(time_steps, callback=callback, log_interval=log_interval)

    if use_wandb_entity:
        run.finish()
    if show:
        plot_training_rewards(model_name)
        plot_evaluation_rewards(model_name)

    return model
