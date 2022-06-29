import numpy as np 


class Heuristic:
    """Decision-rule based agent. Use deterministic approximation to hire the
    amount that gets as close to the goal state as possible.

    :param env: Environment class (like in predefined_models). Currently only
        MultiDiscrete-based models are supported (it requires a hire_options 
        attribute).
    """
    def __init__(self, env):
        self.env = env
        self.max_hire_amounts = [opt[-1] for opt in self.env.hire_options]
        
    def predict(self, state, deterministic=True):
        """Determine optimal actions. Same format as predict methods in RL 
        agents, so has deterministic parameter, but always outputs a 
        deterministic policy anyway.

        :param state: Observation of the current state for which the best
            action has to be chosen.
        :param deterministic: Dummy bool in order to have this method in the
            same format as the predict method in RL agent classes.
        :return: The calculated action, mapped to the closest discrete 
            options within the hire_options of the environment, as well as the
            unmapped calculated action (this also makes sure that the method
            returns two values, like the other predict methods).
        """
        hires = []
        for i in range(self.env.n_cohorts):
            next_state = np.dot(
                self.env._state_to_headcount(state), self.env.p_matrix[:, i])
            if next_state <= self.env.goal_state[i]:
                action = min(self.env.goal_state[i] - next_state, 
                             self.max_hire_amounts[i])
            else: 
                action = -min(next_state - self.env.goal_state[i], 
                              self.max_hire_amounts[i])
            hires.append(action)
        # Map policy to multi-discrete variant
        time_step_closest_actions = [
            np.argmin([
                abs(cohort_action - hire_option) 
                for hire_option in self.env.hire_options[i]])
            for i, cohort_action in enumerate(hires)]
        return time_step_closest_actions, hires
