"""
Adaptation of the MultiCategorialDistribution class developed by stable-baselines3.
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py
"""
from typing import List, Tuple

import torch as th
from stable_baselines3.common.distributions import Distribution
from torch import nn
from torch.distributions import Categorical


class MultiCategoricalDistribution(Distribution):
    """
    MultiCategorical distribution for multi discrete actions.

    Adapted from stable-baselines3: added action_quality method.

    :param action_dims: List of sizes of discrete action spaces
    """
    def __init__(self, action_dims: List[int]):
        super(MultiCategoricalDistribution, self).__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """

        action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "MultiCategoricalDistribution":
        self.distribution = [Categorical(logits=split) for split in th.split(action_logits, tuple(self.action_dims), dim=1)]
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        # Extract each discrete action and compute log prob for their respective distributions
        return th.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, th.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self) -> th.Tensor:
        return th.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) -> th.Tensor:
        return th.stack([dist.sample() for dist in self.distribution], dim=1)

    def action_quality(self):
        chosen_probs = []
        for i in range(len(self.distribution)):
            probs = [dist.probs for dist in self.distribution][i][0]
            prob_of_chosen_action = probs[th.argmax(probs)]
            chosen_probs.append(prob_of_chosen_action)
        return chosen_probs

    def mode(self) -> th.Tensor:
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob
