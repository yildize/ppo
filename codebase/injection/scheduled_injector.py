import gym
import torch

from injection.assistive_actors.factory import AssitiveActorFactory
from injection.assistive_actors.mountaincar import BaseMountainCarAssistiveActor
from injection.base_action_injector import BaseActionInjector
from injection.scheludes import ready_schedules
from injection.scheludes.injection_schedule import InjectionSchedule
from injection.scheludes.ready_schedules import InjectionSchedules
from utils.enums import NoiseUpdateFreq
from utils.render_wrapper import RenderWrapper
from utils.utils import MultivariateGaussianDist
from typing import Tuple, List, Union


class ScheduledInjector(BaseActionInjector):
    """This a bit more advanced injector class that helps providing more complex injecting patterns with the help
    of a schedule."""

    assistive_actor: BaseMountainCarAssistiveActor

    def __init__(self, actor: torch.nn.Module, multivariate_gauss_dist: MultivariateGaussianDist, total_timesteps: int, episode_counter: List[int],
                 schedule_enum: InjectionSchedules, env: Union[gym.Env, RenderWrapper], verbose=False):
        super().__init__(actor=actor, multivariate_gauss_dist=multivariate_gauss_dist, total_timesteps=total_timesteps, episode_counter=episode_counter)
        self.__env = env
        self.__prev_episode = -1
        self.__prev_period = None
        self.schedule: InjectionSchedule = schedule_enum.value

        self._noise = None
        self._noise_update_freq = None
        self._noise_std = None

        # Log purposes
        self.__verbose = verbose
        self.__prev_noise = None
        self.__prev_assist_w = None

    def inject_action(self, state, current_timestep) -> Tuple:
        # First check if there is a current injection:
        active_period = self.schedule.get_current_injection(current_timestep)
        if active_period is not None:
            if self.__prev_period != active_period:  # If a new period is activated
                print("New injection period activated: ", active_period)
                self.assistive_actor = AssitiveActorFactory.create(active_period.assistive_actor_type)
                self._noise_update_freq, self._noise_std = active_period.noise_update_freq, active_period.noise_std

            assist_w_raw = self.schedule.get_current_w(active_period, current_timestep)
            self.__update_noise() # updates self._noise depending on the update_freq
            assist_w = torch.clip(assist_w_raw + self._noise, min=active_period.min_w, max=active_period.max_w)
            recommended_action = self.assistive_actor.get_action(state=state)
            actor_action, action_dist = self.sample_actor_action(state=state)
            action = assist_w * recommended_action + (1 - assist_w) * actor_action
            self.__prev_assist_w = assist_w
        else: # No active period, thus act only considering the agent itself.
            if self.__prev_period is not None and active_period is None:
                print("Assistant schedule period is ended, control is on the Agent!")
            action, action_dist = self.sample_actor_action(state=state)

        self.__prev_period = active_period
        self.__prev_episode = self.current_episode
        self.__prev_noise = self._noise

        # action = torch.clip(action, min=-1, max=1)
        log_prob_a = action_dist.log_prob(action)

        return action.numpy(), log_prob_a.item()

    def __update_noise(self,):
        """ This method is used to update the noise , typically I want to update the noise for each
        episode or for each rollout."""
        if self._noise_update_freq == NoiseUpdateFreq.every_episode:
            if self.current_episode != self.__prev_episode:
                self._noise = torch.randn((1,))*self._noise_std
        elif self._noise_update_freq == NoiseUpdateFreq.every_action:
            self._noise = torch.randn((1,)) * self._noise_std
        else:
            ...

        if self.__verbose and self.__prev_noise != self._noise:
            print(f"Noise updated from {self.__prev_noise} to {self._noise}")




