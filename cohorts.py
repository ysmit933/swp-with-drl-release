"""
Main cohort models. 
"""
from abc import ABC, abstractmethod

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from helpers import continuous_multinomial


MIN_COHORT_SIZE = 0
MAX_COHORT_SIZE = 10_000


class Cohorts(gym.Env, ABC):
    """Base class of the cohorts setup. 
    
    The observation space and dynamics are determined by transition matrix P.
    The action space is specified in the different sub-classes (continuous / multi-discrete).

    :param p_matrix: Matrix of probabilities of transitioning between cohorts.
        Needs to be provided without a separate attrition column.
    :param reward: Reward function class. (WeightedReward or ConstrainedReward)
    :param time_horizon: Number of time-steps the environment can run. Default: 50
    :param max_cohort_sizes: List of maximum capacity of each cohort. 
        Observations will be clipped to these values.
        Also used to normalize the state space. Defaults to 10.000 when none provided.
    :param min_cohort_sizes: List of minimum capacity of each cohort. 
        Observations will be clipped to these values. Default to 0 when none provided.
    :param salary_costs: List of salary costs for each cohort.
    :param hiring_costs: List of hiring costs for each cohort.
    :param starting_state: Cohort levels that the environment start in. 
        Needs to be provided as a list of headcounts.
    :param random_start_percentage: Level of randomness of generated random starting states.
    :param random_start_around_fixed: Whether the generated random starting states 
        are centered around the fixed starting state or between 0 and max_cohort_sizes.
    """
    def __init__(self, p_matrix, reward, time_horizon=60, 
            max_cohort_sizes=None, min_cohort_sizes=None, salary_costs=None, 
            hiring_costs=None, starting_state=None, firing_costs=None,
            random_start_percentage=0.0, random_start_around_fixed=False,
            early_termination=False, reward_n_enough=5, reward_threshold=0.95, 
            reward_std=0.01, completely_random_starting_state=False
    ):
        super(Cohorts, self).__init__()
        self.n_cohorts = len(p_matrix[0])
        self.reward = reward
        self.time_horizon = time_horizon
        self.salary_costs = salary_costs
        self.hiring_costs = hiring_costs
        self.firing_costs = firing_costs or hiring_costs
        self.early_termination = early_termination
        self.starting_state = np.array(starting_state)
        self.random_start_percentage = random_start_percentage
        self.random_start_around_fixed = random_start_around_fixed
        self.completely_random_starting_state = completely_random_starting_state

        # Track rewards for termination
        self.rewards = []
        self.reward_n_enough = reward_n_enough
        self.reward_threshold = reward_threshold
        self.reward_std = reward_std

        self.min_cohort_sizes = min_cohort_sizes \
            or np.array([MIN_COHORT_SIZE] * self.n_cohorts) 
        self.max_cohort_sizes = max_cohort_sizes \
            or np.array([MAX_COHORT_SIZE] * self.n_cohorts) 

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_cohorts,), dtype=np.float64)

        # To make sure the rows sum to one (for the multinomial 
        # distribution to work), create one more column in the transition 
        # matrix that represents the attrition state.
        self.p_matrix = np.concatenate((
            p_matrix, 
            np.array(
                [1 - sum(p_matrix[i,:]) for i in range(self.n_cohorts)]
            ).reshape(self.n_cohorts, 1)
        ), axis=1)

    def step(self, action):
        """Perform action and advance the environment accordingly for one time-step.

        The use of methods action_to_headcount, state_to_headcount and 
        headcount_to_state ensures that this general calculation of the dynamics 
        can be used in all different (continuous/multi-discrete/constrained firing) cohort models.

        :param action: The action to take in the environment.
        :return: The environment after taking the action, the reward obtained, 
            whether or not the episode is over, and some additional info used for logging/debugging. 
        """
        # Remember current state to calculate reward
        old_state = self.state

        # Update state according to the action and the dynamics
        hires = self._action_to_headcount(action)
        movers_from_state = np.concatenate([
            continuous_multinomial(cohort, self.p_matrix[i,:]).reshape(1, -1) 
            for i, cohort in enumerate(self._state_to_headcount(self.state))], axis=0)[:,:-1]
        new_headcount_state = np.clip(
            np.sum(movers_from_state, axis=0) + hires, 
            self.min_cohort_sizes, self.max_cohort_sizes)
        self.state = self._headcount_to_state(new_headcount_state, action, old_state)

        # Calculate reward
        current_reward = self.reward.calculate(old_state, action, self.state)
        self.rewards.append(current_reward)

        # Termination conditions
        self.n_steps += 1
        at_time_horizon = self.n_steps >= self.time_horizon
        good_enough = self.n_steps > self.reward_n_enough and np.mean(self.rewards[-self.reward_n_enough:]) > self.reward_threshold
        no_change = self.n_steps > self.reward_n_enough and np.std(self.rewards[-self.reward_n_enough:]) < self.reward_std
        if self.early_termination:
            done = at_time_horizon or good_enough or no_change
        else:
            done = at_time_horizon

        info = dict(
            reward_info=self.reward.get_info(old_state, action, self.state), 
            termination=dict(time_horizon=at_time_horizon, good_enough=good_enough, no_change=no_change)
        )

        return self.state, current_reward, done, info

    def reset(self, state=None):
        """Reset the environment. 
        
        Resets to a specific state if that is provided, otherwise resets to a 
        generated random starting state.

        :param state: State to reset to. Resets to a randomly generated state when none provided.
        :return: The state where the environment has reset to.
        """
        if state is not None:
            self.state = self._headcount_to_state(state) 
        else:
            self.state = self._random_starting_state()
        self.n_steps = 0
        self.rewards = []
        return self.state

    def calculate_soc(self):
        """Calculate current (average) span-of-control by dividing the total 
        number of contributors by the total number of managers. 
        
        This method assumes that the first half of cohorts are managers 
        and the second half are contributors, which is the case for all used models.
        """
        headcounts = self.headcounts()
        num_managers = sum(headcounts[:self.n_cohorts // 2])
        num_contributors = sum(headcounts[self.n_cohorts // 2:self.n_cohorts])
        return 0 if num_managers == 0 else num_contributors / num_managers

    def current_salary_costs(self, total=True):
        """Calculate the (monthly) salary costs for the current state.

        Returns the total over all cohorts if total parameter is set to True,
        otherwise returns the salary costs per cohort.
        """
        if not self.salary_costs:
            raise ValueError('No salary costs provided.')
        if total:
            return np.dot(self.headcounts(), self.salary_costs)
        else:
            return np.multiply(self.headcounts(), self.salary_costs)

    def current_hiring_costs(self, action, total=True):
        """Calculate the (monthly) hiring costs of the action taken.

        Returns the total over all cohorts if total parameter is set to True,
        otherwise returns the hiring costs per cohort.
        """
        # Also ensures there are firing costs provided
        if not self.hiring_costs:
            raise ValueError('No hiring costs provided.')
        costs = [ 
            -1 * a * self.firing_costs[i] if a < 0 else a * self.hiring_costs[i]
            for i, a in enumerate(self._action_to_headcount(action))
        ]
        return sum(costs) if total else costs

    def current_employee_count(self):
        """Calculate the current total headcount of the environment."""
        return np.sum(self.headcounts())

    def headcounts(self):
        """Returns the headcount of each cohort."""
        return self._state_to_headcount(self.state)

    def show_p_matrix(self, precision=2):
        """Print probability transition matrix in a more readable format."""
        print(np.around(self.p_matrix, precision))

    def show_p_matrix_code(self, precision=4):
        """Print probability transition matrix in a way that can be copied and
        adjusted in code.
        """
        for r in self.p_matrix:
            print("[" + ",".join([str(round(x, precision)) for x in r[:-1]]) + "],")

    def seed(self, seed=None):
        """For OpenAI Gym."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        """This method is called by wrappers like Monitor."""
        pass        

    @abstractmethod
    def _random_starting_state(self):
        """To be implemented based on continuous/discrete action space and/or the 
        possibility of firing.
        """
    
    @abstractmethod 
    def _state_to_headcount(self, state):
        """To be implemented based on continuous/discrete action space and/or the 
        possibility of firing.
        """
    
    @abstractmethod 
    def _headcount_to_state(self, headcount, action=None, old_state=None):
        """To be implemented based on continuous/discrete action space and/or the 
        possibility of firing.

        Includes an action/old_state parameter, since the model including firing 
        needs to store this in the state.
        """

    @abstractmethod 
    def _action_to_headcount(self, action):
        """To be implemented based on continuous/discrete action space and/or the 
        possibility of firing.
        """


class MultiDiscreteCohorts(Cohorts):
    """Version of the cohort model where the possible hiring amounts are fixed 
    to a set of discrete values. This also makes the observations 
    multi-discrete, but the state space is still normalized (to [0,1]) using 
    the maximum cohort sizes.

    :param p_matrix: Matrix of probabilities of transitioning between cohorts.
        Needs to be provided without a separate attrition column.
    :param reward: Reward function class. (WeightedReward or ConstrainedReward)
    :param hire_options: List of lists with possible hiring options per cohort.
        Negative values are allowed (representing firing). Note: current 
        implementation assumes that every list of options contains the value 0
        and that, when negative values are included, there are an equal number 
        of positive and negative values. For example: 
        [[-2, -1, 0, 1, 2], [-5, -1, 0, 1, 5]].
    :param kwargs: Additional keyword arguments to be passed to the main cohort
        model.
    """
    def __init__(self, p_matrix, reward, hire_options, **kwargs):
        super(MultiDiscreteCohorts, self).__init__(p_matrix, reward, **kwargs)
        self.hire_options = hire_options
        self.action_space = spaces.MultiDiscrete([len(options) for options in self.hire_options])

    def _state_to_headcount(self, state):
        # Cast the values to int to account for small errors caused by dividing and multiplying the states
        return [int(np.round(x)) for x in np.multiply(state, self.max_cohort_sizes)]
    
    def _headcount_to_state(self, headcount, action=None, old_state=None):
        return np.divide(headcount, self.max_cohort_sizes)

    def _action_to_headcount(self, action):
        return [self.hire_options[i][a] for i, a in enumerate(action)]

    def _random_starting_state(self):
        if self.completely_random_starting_state:
            return self.observation_space.sample()

        if self.random_start_around_fixed:
            # Based around provided starting state
            lb = [int((1 - self.random_start_percentage) * x) for x in self.starting_state]
            ub = [int((1 + self.random_start_percentage) * x) for x in self.starting_state]
        else:
            # Based on max cohort sizes
            lb = [int((1 - self.random_start_percentage) / 2 * m) for m in self.max_cohort_sizes]
            ub = [int((1 + self.random_start_percentage) / 2 * m) for m in self.max_cohort_sizes]
        random_state = np.array([l if l == b else np.random.randint(l, b) for l, b in zip(lb, ub)])
        return self._headcount_to_state(random_state)


class MultiDiscreteFiringWindowCohorts(MultiDiscreteCohorts):
    """Extension of the multi-discrete cohort model to constrain firing. 
    
    For each cohort is being tracked whether it there are previous hires or 
    fires. If there were any hires (resp. fires) in the last time_window 
    number of time-steps, then only new hires (resp. fires) or no new hires 
    (resp. fires) are allowed. The constraint is lifted when there were no 
    new hires/fires for time_window number of time-steps.

    :param p_matrix: Matrix of probabilities of transitioning between cohorts.
        Needs to be provided without a separate attrition column.
    :param reward: Reward function class. (WeightedReward or ConstrainedReward)
    :param firing_costs: Costs of a "negative hire" per cohort. Can be used in
        additional reward functions.
    :param time_window: The number of time-steps after which the constraint is 
        lifted if the agent doesn't hire of fire anyone for this amount of 
        time-steps.
    :param kwargs: Additional keyword arguments to be passed to the main cohort
        model.
    """
    def __init__(self, p_matrix, reward, firing_costs=None, time_window=6, **kwargs):
        super(MultiDiscreteFiringWindowCohorts, self).__init__(p_matrix, reward, **kwargs)
        self.firing_costs = firing_costs
        self.time_window = time_window

        # State space is extended with n_cohort variables that describe the allowed actions
        # and another n_cohort timer variables that keep track of when to lift the constraint.
        # All variables are normalized between [0,1] to match the rest of the state space.
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_cohorts * 3,))

    def _state_to_headcount(self, state):
        return np.multiply(state[:self.n_cohorts], self.max_cohort_sizes)
    
    def _headcount_to_state(self, headcount, action=None, old_state=None):
        if action is None or old_state is None:
            # Starting state
            normalized_action = np.concatenate((0.5 * np.ones(self.n_cohorts), np.zeros(self.n_cohorts)))
        else:
            normalized_action = self._action_to_state(action, old_state)
        return np.concatenate((np.divide(headcount, self.max_cohort_sizes), normalized_action))

    def _action_to_state(self, action, old_state):
        """This method now also needs to calculate the additional flag states 
        in the state space that manage the firing constraints.
        """
        if self.time_window == 1:
            counter = 1
        else:
            counter = 1 / (self.time_window - 1)
        flags = old_state[self.n_cohorts:]
        additional_flag_states = np.zeros(self.n_cohorts * 2)
        normalized_action = self._action_to_headcount(action)
        for i in range(self.n_cohorts):
            a = normalized_action[i]
            if a > 0:
                new_flag = 1
                timer = 0
            elif a < 0:
                new_flag = 0
                timer = 0
            else:
                current_timer = flags[self.n_cohorts+i]
                if flags[i] == 0.5:
                    new_flag = 0.5
                    timer = 0
                elif flags[i] == 1:
                    if current_timer == 1 or self.time_window == 1:
                        new_flag = 0.5
                        timer = 0
                    else:
                        new_flag = 1
                        timer = current_timer + counter
                elif flags[i] == 0:
                    if current_timer == 1 or self.time_window == 1:
                        new_flag = 0.5
                        timer = 0
                    else:
                        new_flag = 0
                        timer = current_timer + counter
            additional_flag_states[i] = new_flag
            additional_flag_states[self.n_cohorts+i] = timer

        return additional_flag_states
        