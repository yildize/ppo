import gym
import torch

from injection.assistive_actors.factory import AssitiveActorFactory
from injection.assistive_actors.lunarlander import LunarLanderAssistiveActor
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
    """This injector module helps realizing complex injection patterns/scenarios through utilizing a InjectionSchedule
    instance which acts like a configuration object encapsulating the details of the injection rouitine."""
    assistive_actor: Union[BaseMountainCarAssistiveActor, LunarLanderAssistiveActor]

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
        """ This is the method that handles the actual injection job. It simply
        accepts the state and the current timestep as an argument, and using this current_timestep info
        and the inner schedule, it can check what exactly should be done (injection or not, if injection
        what are the details, how to calculate the weights, which assistant should be used, ...)"""

        # First check if there is a current injection:
        active_period = self.schedule.get_current_injection(current_timestep)
        if active_period is not None:  # If a new period is activated
            if self.__prev_period != active_period:  # If a new period is activated
                print("New injection period activated: ", active_period)
                # Get the assitive actor:
                self.assistive_actor = AssitiveActorFactory.create(active_period.assistive_actor_type)
                self._noise_update_freq, self._noise_std = active_period.noise_update_freq, active_period.noise_std

            # For the active period, get the assistant action weight using the schedule:
            assist_w_raw = self.schedule.get_current_w(active_period, current_timestep)
            # Update the weight noise according to schedule configuration
            self.__update_noise() # updates self._noise depending on the update_freq
            # Apply the current noise to the raw assistant action weight and clip it:
            assist_w = torch.clip(assist_w_raw + self._noise, min=active_period.min_w, max=active_period.max_w)
            # Get the recommended/assistive action from assitive actor:
            recommended_action = self.assistive_actor.get_action(state=state)
            # Get the actual actor action and the current action distribution (utilizing actor network)
            actor_action, action_dist = self.sample_actor_action(state=state)
            # Now construct the resultant action via weighted sum:
            action = assist_w * recommended_action + (1 - assist_w) * actor_action
            self.__prev_assist_w = assist_w
        else: # No active period, thus act only considering the agent itself.
            if self.__prev_period is not None and active_period is None:
                print("Assistant schedule period is ended, control is on the Agent!")
            # Just get the regular actor action using the actor:
            action, action_dist = self.sample_actor_action(state=state)

        self.__prev_period = active_period
        self.__prev_episode = self.current_episode
        self.__prev_noise = self._noise

        # Pay attention, log prob is calculated using the resultant action, so the agent would think
        # It took the action completely by itself.
        log_prob_a = action_dist.log_prob(action) # action = torch.clip(action, min=-1, max=1)

        return action.numpy(), log_prob_a.item()

    def __update_noise(self,):
        """ This method is used to update the action weight noise, this noise can be updated
        at each action or at each episode depending on the configuration."""
        if self._noise_update_freq == NoiseUpdateFreq.every_episode:
            if self.current_episode != self.__prev_episode:
                self._noise = torch.randn((1,))*self._noise_std
        elif self._noise_update_freq == NoiseUpdateFreq.every_action:
            self._noise = torch.randn((1,)) * self._noise_std
        else:
            ...

        if self.__verbose and self.__prev_noise != self._noise:
            print(f"Noise updated from {self.__prev_noise} to {self._noise}")




