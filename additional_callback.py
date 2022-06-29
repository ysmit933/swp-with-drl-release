from stable_baselines3.common.callbacks import BaseCallback
import torch as th

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record('buffer_values', self.locals['rollout_buffer'].values.mean().item())
        values = th.Tensor(self.locals['rollout_buffer'].values.flatten())
        early_term = int(self.locals['infos'][0]['termination']['good_enough'])
        self.logger.record('early_termination', early_term)

        early_term = int(self.locals['infos'][0]['termination']['no_change'])
        self.logger.record('early_no_change', early_term)

        values, log_probs, entropy = self.model.policy.evaluate_actions(self.locals['obs_tensor'], th.Tensor(self.locals['actions']))

        self.logger.record('estimated_value', values.mean().item())
        self.logger.record('log_probs', log_probs.mean().item())
        self.logger.record('entropy', entropy[0].item())
        self.logger.record('predicted_values', self.locals['rollout_buffer'].values.mean())
        self.logger.record('empirical_return', self.locals['rollout_buffer'].returns.mean())
        
        return True
