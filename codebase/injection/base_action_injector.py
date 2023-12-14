from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch.distributions import MultivariateNormal

from injection.assistive_actors.factory import AssitiveActorFactory
from utils.enums import AssitiveActors
from utils.utils import MultivariateGaussianDist


class BaseActionInjector(ABC):
    """ This is the base class that will be inherited by action injectors. This class should provide
    required objects and methods, also it should provide an interface to be followed."""

    def __init__(self, actor:torch.nn.Module, multivariate_gauss_dist:MultivariateGaussianDist, total_timesteps:int, assitive_actor:AssitiveActors):
        self.actor = actor
        self.multivariate_gauss_dist = multivariate_gauss_dist
        self.total_timesteps = total_timesteps
        self.assistive_actor = AssitiveActorFactory.create(assitive_actor)
        # noise
        # injection strategy

    @abstractmethod
    def inject(self, state, current_timestep)->Tuple:
        ...
        # should return action and the log_prob_a

    def completed_training_frac(self, current_timestep):
        return 1-((self.total_timesteps-current_timestep)/self.total_timesteps)

    def sample_actor_action(self, state):
        action_dist = self.get_action_dist()
        actor_action = action_dist.sample()
        return actor_action

    def get_action_dist(self, state) -> MultivariateNormal:
        """This method will return the action distribution for the provided state"""
        mean, std = self.get_actor_mean_std(state=state)
        action_dist = self.multivariate_gauss_dist.gaussian_action_dist(mean=mean, std=std)
        return action_dist

    def get_actor_mean_std(self, state):
        """ Just returns the actor action (the mean) for the provided state."""
        with torch.no_grad():
            res = self.actor(state)

            if isinstance(res, tuple): mean, std = res
            else: mean, std = res, None

            # this will be the mean probs vector of shape (num_actions,)
            return mean, std