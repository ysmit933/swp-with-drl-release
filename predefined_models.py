"""
Cohort models used for the experiments described in the paper. 

Settings and parameters can be adjusted by setting the variables at the
top of this file. 

Parameters can also be set when defining an instance of the model using 
keyword arguments.
"""
from lib2to3.pgen2.token import OP
import math

import numpy as np
import pandas as pd

from cohorts import *
from rewards import *


# Set path to Excel file containing transition probability data
P_MATRIX_EXCEL_FILE = 'example_transition_probability_matrix.xlsx'  

# Load transition probability data
df = pd.read_excel(f'{P_MATRIX_EXCEL_FILE}')
df.drop('Unnamed: 0', axis='columns', inplace=True)
P_MATRIX = np.nan_to_num(df)
N_COHORTS = len(P_MATRIX)

# Starting state headcounts
STARTING_STATE = [100] * N_COHORTS

# Target state headcounts
# Either as objective for operational reward, or as target for heuristic 
# applied to strategic reward (requires calculated optimal state that 
# yields the highest immediate reward).
TARGET_STATE = [100] * N_COHORTS

# Set custom hiring options
HIRE_OPTIONS = [[-1, 0, 1]] * N_COHORTS

# Model assumes that first n // 2 cohorts are Managers
# and the last n // 2 cohorts are Contributors
SALARY_COSTS = [5_000.00] * (N_COHORTS // 2) + [2_000.00] * (N_COHORTS // 2)

# tau parameter for constrained firing (mask and penalty) environments
TIME_WINDOW_CONSTRAINED_FIRING = 5

# Reward functions parameters
ALPHA = 1.0
COHORT_LB = 0.75
COHORT_UB = 1.25
TARGET_SOC = 7
SOC_LB = 0.9


# Case study models
class CaseStudyCombined(MultiDiscreteCohorts):
    def __init__(self, p_matrix=None, starting_state=None, goal_state=None, 
            hire_options=None, **kwargs):
        if p_matrix is None:
            df = pd.read_excel(f'{P_MATRIX_EXCEL_FILE}')
            df.drop('Unnamed: 0', axis='columns', inplace=True)
            p_matrix = np.nan_to_num(df)
        
        starting_state = starting_state or STARTING_STATE
        self.goal_state = goal_state or TARGET_STATE
        
        hire_options = HIRE_OPTIONS
        self.max_hires = [option[-1] for option in hire_options]
        max_cohort_sizes = [max(1, 2*x) for x in starting_state]

        reward_specs = {
            "method": "combined",
            "lb": [math.floor(COHORT_LB * x) for x in starting_state],
            "ub": [math.ceil(COHORT_UB * x) for x in starting_state],
            "soc": TARGET_SOC,
            "soc_lb": SOC_LB,
            "salary": SALARY_COSTS
        }
        reward = StrategicReward(self, reward_specs)
        super(CaseStudyCombined, self).__init__(
            p_matrix = p_matrix, 
            reward = reward,
            max_cohort_sizes = max_cohort_sizes,
            salary_costs = SALARY_COSTS,
            hire_options = HIRE_OPTIONS,
            starting_state = starting_state,
            **kwargs
        )


class CaseStudyCombinedNoFire(MultiDiscreteCohorts):
    def __init__(self, p_matrix=None, starting_state=None, goal_state=None, 
            hire_options=None, **kwargs):
        if p_matrix is None:
            df = pd.read_excel(f'{P_MATRIX_EXCEL_FILE}')
            df.drop('Unnamed: 0', axis='columns', inplace=True)
            p_matrix = np.nan_to_num(df)
        
        starting_state = starting_state or STARTING_STATE
        self.goal_state = goal_state or TARGET_STATE
        
        # Only use positive hire amounts
        hire_options = [
            [hire_amount for hire_amount in options if hire_amount >= 0] 
            for options in HIRE_OPTIONS
        ]
        self.max_hires = [option[-1] for option in hire_options]
        max_cohort_sizes = [max(1, 2*x) for x in starting_state]

        reward_specs = {
            "method": "combined",
            "lb": [math.floor(COHORT_LB * x) for x in starting_state],
            "ub": [math.ceil(COHORT_UB * x) for x in starting_state],
            "soc": TARGET_SOC,
            "soc_lb": SOC_LB,
            "salary": SALARY_COSTS
        }
        reward = StrategicReward(self, reward_specs)
        super(CaseStudyCombinedNoFire, self).__init__(
            p_matrix = p_matrix, 
            reward = reward,
            max_cohort_sizes = max_cohort_sizes,
            salary_costs = SALARY_COSTS,
            hire_options = hire_options,
            starting_state = starting_state,
            **kwargs
        )


class CaseStudyCombinedMask(MultiDiscreteFiringWindowCohorts):
    def __init__(self, p_matrix=None, starting_state=None, goal_state=None, 
            hire_options=None, time_window=TIME_WINDOW_CONSTRAINED_FIRING, **kwargs):
        if p_matrix is None:
            df = pd.read_excel(f'{P_MATRIX_EXCEL_FILE}')
            df.drop('Unnamed: 0', axis='columns', inplace=True)
            p_matrix = np.nan_to_num(df)
        
        starting_state = starting_state or STARTING_STATE
        self.goal_state = goal_state or TARGET_STATE
        
        hire_options = HIRE_OPTIONS
        self.max_hires = [option[-1] for option in hire_options]
        max_cohort_sizes = [max(1, 2*x) for x in starting_state]

        reward_specs = {
            "method": "combined",
            "lb": [math.floor(COHORT_LB * x) for x in starting_state],
            "ub": [math.ceil(COHORT_UB * x) for x in starting_state],
            "soc": TARGET_SOC,
            "soc_lb": SOC_LB,
            "salary": SALARY_COSTS
        }
        reward = StrategicReward(self, reward_specs)
        super(CaseStudyCombinedMask, self).__init__(
            p_matrix = p_matrix, 
            reward = reward,
            max_cohort_sizes = max_cohort_sizes,
            salary_costs = SALARY_COSTS,
            hire_options = HIRE_OPTIONS,
            starting_state = starting_state,
            time_window = time_window,
            **kwargs
        )


class CaseStudyCombinedPenalty(MultiDiscreteFiringWindowCohorts):
    def __init__(self, p_matrix=None, starting_state=None, hire_options=None, 
            goal_state=None, time_window=TIME_WINDOW_CONSTRAINED_FIRING, **kwargs):
        if p_matrix is None:
            df = pd.read_excel(f'{P_MATRIX_EXCEL_FILE}')
            df.drop('Unnamed: 0', axis='columns', inplace=True)
            p_matrix = np.nan_to_num(df)
        
        starting_state = starting_state or STARTING_STATE
        self.goal_state = goal_state or TARGET_STATE
        
        hire_options = HIRE_OPTIONS
        self.max_hires = [option[-1] for option in hire_options]
        max_cohort_sizes = [max(1, 2*x) for x in starting_state]

        reward_specs = {
            "method": "combined",
            "lb": [math.floor(COHORT_LB * x) for x in starting_state],
            "ub": [math.ceil(COHORT_UB * x) for x in starting_state],
            "soc": TARGET_SOC,
            "soc_lb": SOC_LB,
            "salary": SALARY_COSTS
        }
        reward = StrategicRewardFiringWindowPenalty(self, reward_specs)
        super(CaseStudyCombinedPenalty, self).__init__(
            p_matrix = p_matrix, 
            reward = reward,
            max_cohort_sizes = max_cohort_sizes,
            salary_costs = SALARY_COSTS,
            hire_options = HIRE_OPTIONS,
            starting_state = starting_state,
            time_window = time_window,
            **kwargs
        )
        
        
class CaseStudyTarget(MultiDiscreteCohorts):
    def __init__(self, p_matrix=None, starting_state=None, hire_options=None, 
            goal_state=None, alpha=ALPHA, **kwargs):
        if p_matrix is None:
            df = pd.read_excel(f'{P_MATRIX_EXCEL_FILE}')
            df.drop('Unnamed: 0', axis='columns', inplace=True)
            p_matrix = np.nan_to_num(df)
        
        starting_state = starting_state or STARTING_STATE
        self.goal_state = goal_state or TARGET_STATE
        
        hire_options = HIRE_OPTIONS
        self.max_hires = [option[-1] for option in hire_options]
        max_cohort_sizes = [max(1, 2*x) for x in starting_state]

        reward_specs = {
            "method": "target",
            "target_state": self.goal_state,
            "alpha": alpha
        }
        reward = OperationalReward(self, reward_specs)
        super(CaseStudyTarget, self).__init__(
            p_matrix = p_matrix, 
            reward = reward,
            max_cohort_sizes = max_cohort_sizes,
            salary_costs = SALARY_COSTS,
            hire_options = HIRE_OPTIONS,
            starting_state = starting_state,
            **kwargs
        )


class CaseStudyTargetNoFire(MultiDiscreteCohorts):
    def __init__(self, p_matrix=None, starting_state=None, hire_options=None, 
            goal_state=None, alpha=ALPHA, **kwargs):
        if p_matrix is None:
            df = pd.read_excel(f'{P_MATRIX_EXCEL_FILE}')
            df.drop('Unnamed: 0', axis='columns', inplace=True)
            p_matrix = np.nan_to_num(df)
        
        starting_state = starting_state or STARTING_STATE
        self.goal_state = goal_state or TARGET_STATE
        
        # Only use positive hire amounts
        hire_options = [
            [hire_amount for hire_amount in options if hire_amount >= 0] 
            for options in HIRE_OPTIONS
        ]
        self.max_hires = [option[-1] for option in hire_options]
        max_cohort_sizes = [max(1, 2*x) for x in starting_state]

        reward_specs = {
            "method": "target",
            "target_state": self.goal_state,
            "alpha": alpha
        }
        reward = OperationalReward(self, reward_specs)
        super(CaseStudyTargetNoFire, self).__init__(
            p_matrix = p_matrix, 
            reward = reward,
            max_cohort_sizes = max_cohort_sizes,
            salary_costs = SALARY_COSTS,
            hire_options = HIRE_OPTIONS,
            starting_state = starting_state,
            **kwargs
        )
        

class CaseStudyTargetMask(MultiDiscreteFiringWindowCohorts):
    def __init__(self, p_matrix=None, starting_state=None, hire_options=None, 
            goal_state=None, time_window=TIME_WINDOW_CONSTRAINED_FIRING, alpha=ALPHA, **kwargs):
        if p_matrix is None:
            df = pd.read_excel(f'{P_MATRIX_EXCEL_FILE}')
            df.drop('Unnamed: 0', axis='columns', inplace=True)
            p_matrix = np.nan_to_num(df)
        
        starting_state = starting_state or STARTING_STATE
        self.goal_state = goal_state or TARGET_STATE
        
        hire_options = HIRE_OPTIONS
        self.max_hires = [option[-1] for option in hire_options]
        max_cohort_sizes = [max(1, 2*x) for x in starting_state]

        reward_specs = {
            "method": "target",
            "target_state": self.goal_state,
            "alpha": alpha
        }
        reward = OperationalReward(self, reward_specs)
        super(CaseStudyTargetMask, self).__init__(
            p_matrix = p_matrix, 
            reward = reward,
            max_cohort_sizes = max_cohort_sizes,
            salary_costs = SALARY_COSTS,
            hire_options = HIRE_OPTIONS,
            starting_state = starting_state,
            time_window = time_window,
            **kwargs
        )


class CaseStudyTargetPenalty(MultiDiscreteFiringWindowCohorts):
    def __init__(self, p_matrix=None, starting_state=None, hire_options=None, 
            goal_state=None, time_window=TIME_WINDOW_CONSTRAINED_FIRING, alpha=ALPHA, **kwargs):
        if p_matrix is None:
            df = pd.read_excel(f'{P_MATRIX_EXCEL_FILE}')
            df.drop('Unnamed: 0', axis='columns', inplace=True)
            p_matrix = np.nan_to_num(df)
        
        starting_state = starting_state or STARTING_STATE
        self.goal_state = goal_state or TARGET_STATE
        
        hire_options = HIRE_OPTIONS
        self.max_hires = [option[-1] for option in hire_options]
        max_cohort_sizes = [max(1, 2*x) for x in starting_state]

        reward_specs = {
            "method": "target",
            "target_state": self.goal_state,
            "alpha": alpha
        }
        reward = OperationalRewardFiringWindowPenalty(self, reward_specs)
        super(CaseStudyTargetPenalty, self).__init__(
            p_matrix = p_matrix, 
            reward = reward,
            max_cohort_sizes = max_cohort_sizes,
            salary_costs = SALARY_COSTS,
            hire_options = HIRE_OPTIONS,
            starting_state = starting_state,
            time_window = time_window,
            **kwargs
        )
