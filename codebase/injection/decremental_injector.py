import torch

from injection.assistive_actors.mountaincar import BaseMountainCarAssistiveActor
from injection.base_action_injector import BaseActionInjector
from utils.enums import AssitiveActors, NoiseUpdateFreq
from utils.utils import MultivariateGaussianDist
from typing import Tuple, List


class DecrementalInjector(BaseActionInjector):
    """ This is an outdated injector. It is mainly used for specific injection strategy (decremental noisy injection) only.
    The scheduled injector is a more advanced version that can handle different strategies. """
    assistive_actor: BaseMountainCarAssistiveActor

    def __init__(self, actor: torch.nn.Module, multivariate_gauss_dist: MultivariateGaussianDist, total_timesteps: int, assitive_actor_type: AssitiveActors,
                 episode_counter: List[int], assist_w_init: float = 0.3, min_assist_w=0.0, assist_duration_perc:float=0.3, decrement_noise_std: float = 0.03,
                 noise_update_freq: NoiseUpdateFreq = NoiseUpdateFreq.every_episode):
        super().__init__(actor=actor, multivariate_gauss_dist=multivariate_gauss_dist, total_timesteps=total_timesteps,
                         assitive_actor_type=assitive_actor_type, episode_counter=episode_counter)
        self.assist_w_init = assist_w_init
        self.min_assist_w = min_assist_w
        self.assist_duration_perc = assist_duration_perc  # 0.7 means after 70% of the training assistance will stop.
        self.decrement_noise_std = decrement_noise_std
        self.prev_episode = -1

        self.noise_update_freq = noise_update_freq
        self.noise = None
        self.assist_w = None

    def inject_action(self, state, current_timestep) -> Tuple:
        """ In returns the required action (can be injected or not) utilizing the current state, current timestep,
        the assistive actor, and the actual actor."""
        recommended_action = self.assistive_actor.get_action(state=state)
        actor_action, action_dist = self.sample_actor_action(state=state)

        assist_w = self.__current_assistant_w(current_timestep)
        self.assist_w = assist_w
        action = assist_w*recommended_action + (1-assist_w)*actor_action
        log_prob_a = action_dist.log_prob(action)
        return action.numpy(), log_prob_a.item()

    def __current_assistant_w(self, current_time_step):
        """ This method will return the weight of assistant actor. Which will start from about 0.3 and noisly reduce
        by time, so by time the agent will have been observed some reward paths and hopefully can take over."""
        if self.completed_training_frac(current_time_step) > self.assist_duration_perc: return 0

        assist_w_pure = self.assist_w_init*(1-(current_time_step/self.__total_assistance_steps()))
        self.__update_noise()
        assist_w_noisy = torch.clip(assist_w_pure + self.noise, min=self.min_assist_w, max=self.assist_w_init)
        return assist_w_noisy

    def __update_noise(self):
        """ This method is used to update the noise , typically I want to update the noise for each
        episode or for each rollout."""
        if self.noise_update_freq == NoiseUpdateFreq.every_episode:
            if self.current_episode != self.prev_episode:
                self.noise = torch.randn((1,))*self.decrement_noise_std
                self.prev_episode = self.current_episode
                print("assist_w", self.assist_w)
        else:
            ...

    def __total_assistance_steps(self):
        return self.total_timesteps*self.assist_duration_perc



