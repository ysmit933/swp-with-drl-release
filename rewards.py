"""
Classes for different reward functions used in the cohort models.
"""
import numpy as np
from helpers import linear_min


def reward_bounds(vals, lb, ub, alpha=1):
    """Function to calculate rewards based on bounds. 

    For each element x in the list vals, a value of 1 is calculated if x is 
    between the provided lower and upper bounds, otherwise a value between 0 
    and 1 (depending on how close it is to the bounds) is calculated, using 
    an exponential function. Then the average of all elements is returned.

    :param vals: Input values of which is calculated whether or not they are 
        in bounds.
    :param lb: List of lower bounds
    :param ub: List of upper bounds
    :param alpha: Shape parameter for smoothing the (sub-)reward function. 
        Alpha=1 is a rather smooth curve, and as alpha approaches zero the 
        bound becomes strict. It is required that alpha > 0.
    """
    assert alpha != 0
    rewards = []
    # Cast a single element to a list if necessary 
    if not(type(vals) == list or type(vals) == np.ndarray):
        vals = [vals]
        lb = [lb]
        ub= [ub]
    # Iterate over elements and calculate sub-rewards
    for i, x in enumerate(vals):
        if x < lb[i]:
            subreward = np.exp(10 / alpha * (x - lb[i]) / lb[i])
        elif lb[i] <= x <= ub[i]:
            subreward = 1
        else:
            subreward = np.exp(-10 / alpha * (x - ub[i]) / ub[i])
        rewards.append(subreward)
    return np.mean(rewards)


def reward_goal(vals, goal, alpha=1):
    """Function to calculate rewards based on specific goal values. 

    For each element x in the list vals, a value between 0 
    and 1 is calculated, depending on how close it is to the goal, using 
    an exponential function. Then the average of all elements is returned.

    :param vals: Input values of which is calculated whether or not they are 
        in bounds.
    :param goal: List of goal values.
    :param alpha: Shape parameter for smoothing the (sub-)reward function. 
        Alpha=1 is a rather smooth curve, and as alpha approaches zero the 
        bound becomes strict. It is required that alpha > 0.
    """
    assert alpha != 0
    rewards = []
    # Cast a single element to a list if necessary
    if not(type(vals) == list or type(vals) == np.ndarray):
        vals = [vals]
        goal = [goal]
    # Iterate over values and calculate sub-rewards
    for i, x in enumerate(vals):
        
        if x < 0:
            subreward = 0
        elif goal[i] == 0:
            subreward = np.exp(-10/alpha * x**2)
        else:
            subreward = np.exp(-10/alpha * (x - goal[i])**2 / goal[i]**2)
        rewards.append(subreward)
    return np.mean(rewards)


class OperationalReward:
    def __init__(self, env, specs):
        self.env = env
        self.specs = specs

    def calculate(self, old_state=None, action=None, new_state=None):
        return reward_goal(self.env.headcounts(), self.specs["target_state"], self.specs["alpha"])

    def get_info(self, old_state=None, action=None, new_state=None):
        return {}


class OperationalRewardFiringWindowPenalty:
    def __init__(self, env, specs):
        self.env = env
        self.specs = specs

    def calculate(self, old_state=None, action=None, new_state=None):
        n = self.env.n_cohorts
        hire_amounts = self.env._action_to_headcount(action)
        penalties = []
        for i in range(n):
            penalty_i = 0
            if old_state[n+i] == 0 and hire_amounts[i] > 0:
                max_hire = self.env.hire_options[i][-1]
                penalty_i = hire_amounts[i] / max_hire / 2
            elif old_state[n+i] == 1 and hire_amounts[i] < 0:
                max_fire = self.env.hire_options[i][0]
                penalty_i = hire_amounts[i] / max_fire / 2
            penalties.append(penalty_i)
        penalty = np.mean(penalties)
        return reward_goal(self.env.headcounts(), self.specs["target_state"], self.specs["alpha"]) - penalty

    def get_info(self, old_state=None, action=None, new_state=None):
        return {}


class StrategicReward:
    def __init__(self, env, specs):
        self.env = env
        self.specs = specs

    def calculate(self, old_state=None, action=None, new_state=None):
        # If out of bounds, get negative reward
        new_state = self.env._state_to_headcount(new_state)
        coh_rew = 0
        for i, s in enumerate(new_state):
            if s < self.specs['lb'][i] or s > self.specs['ub'][i]:
                coh_rew -= 1

        # If within bounds, get SoC reward (in [0, 1])  
        n = len(new_state)
        if sum(new_state[:n//2]) == 0:
            cur_soc = 0
        else:
            cur_soc = sum(new_state[n//2:]) / sum(new_state[:n//2])
        # cur_soc = self.env.calculate_soc()
        soc_rew = reward_goal(cur_soc, self.specs['soc']) * 1

        coh_rew += soc_rew
        # SoC reward high enough (>0.9), add salary minimization reward
        sal_rew = linear_min(np.dot(new_state, self.specs['salary']), np.dot(self.specs['lb'], self.specs['salary']), np.dot(self.specs['ub'], self.specs['salary']))
        if soc_rew > self.specs['soc_lb']:
            coh_rew += sal_rew * 1
        
        return coh_rew

    def get_info(self, old_state=None, action=None, new_state=None):
        return {}


class StrategicRewardFiringWindowPenalty:
    def __init__(self, env, specs):
        self.env = env
        self.specs = specs

    def calculate_reward(self, old_state=None, action=None, new_state=None):
        # If out of bounds, get negative reward
        new_state = self.env._state_to_headcount(new_state)
        coh_rew = 0
        for i, s in enumerate(new_state):
            if s < self.specs['lb'][i] or s > self.specs['ub'][i]:
                coh_rew -= 1

        # If within bounds, get SoC reward (in [0, 1])  
        n = len(new_state)
        if sum(new_state[:n//2]) == 0:
            cur_soc = 0
        else:
            cur_soc = sum(new_state[n//2:]) / sum(new_state[:n//2])
        # cur_soc = self.env.calculate_soc()
        soc_rew = reward_goal(cur_soc, self.specs['soc'])

        coh_rew += soc_rew
        # SoC reward high enough (>0.9), add salary minimization reward
        sal_rew = linear_min(np.dot(new_state, self.specs['salary']), np.dot(self.specs['lb'], self.specs['salary']), np.dot(self.specs['ub'], self.specs['salary']))
        if soc_rew > self.specs['soc_lb']:
            coh_rew += sal_rew
        
        return coh_rew

    def calculate(self, old_state=None, action=None, new_state=None):
        n = self.env.n_cohorts
        hire_amounts = self.env._action_to_headcount(action)
        penalties = []
        for i in range(n):
            penalty_i = 0
            if old_state[n+i] == 0 and hire_amounts[i] > 0:
                max_hire = self.env.hire_options[i][-1]
                penalty_i = hire_amounts[i] / max_hire / 2
                penalty_i = 1 #/ self.env.n_cohorts 
            elif old_state[n+i] == 1 and hire_amounts[i] < 0:
                max_fire = self.env.hire_options[i][0]
                penalty_i = hire_amounts[i] / max_fire / 2
                penalty_i = 1 #/ self.env.n_cohorts 
            penalties.append(penalty_i)

        penalty = np.sum(penalties)
        return self.calculate_reward(old_state, action, new_state) - penalty

    def get_info(self, old_state=None, action=None, new_state=None):
        return {}
