from dataclasses import dataclass
from typing import List, Optional

from injection.assistive_actors.factory import AssitiveActorFactory
from utils.enums import AssitiveActors, NoiseUpdateFreq


class InjectionSchedule:
    """ This class helps specifying the injection schedule, lile from which step to
    which step, with which weight and noise, and other details of injection. It is basically
    a helper configuration class."""

    @dataclass
    class InjectionPeriod:
        assistive_actor_type:AssitiveActors
        step_start:int
        step_end:int
        assist_w_start:float = 0.3
        assist_w_end:float = 0.0
        min_w:float = 0,
        max_w:float = 0.3
        noise_std:float = 0.1
        noise_update_freq:NoiseUpdateFreq = NoiseUpdateFreq.every_episode
        render:bool = False

    def __init__(self, periods:List[InjectionPeriod] = None):
        self.injection_periods:List[InjectionSchedule.InjectionPeriod] = [] if periods is None else periods

    def add_a_period(self,  assistive_actor_type:AssitiveActors, step_start:int, step_end:int, assist_w_start:float=0.3, assist_w_end:float=0.0, noise_std:float=0.1, noise_update_freq:NoiseUpdateFreq=NoiseUpdateFreq.every_episode):
        self.injection_periods.append(InjectionSchedule.InjectionPeriod(assistive_actor_type, step_start, step_end, assist_w_start, assist_w_end, noise_std, noise_update_freq))

    def get_current_injection(self, timestep)->Optional[InjectionPeriod]:
        for period in self.injection_periods:
            if period.step_start <= timestep <= period.step_end:
                return period
        return None

    def get_assistive_actor(self, period:InjectionPeriod):
        AssitiveActorFactory.create(assitive_actor=period.assistive_actor_type)

    def get_current_w(self, period:InjectionPeriod, current_timestep):
        if period.step_start <= current_timestep <= period.step_end:
            total_steps = period.step_end - period.step_start
            slope = (period.assist_w_end - period.assist_w_start) / total_steps
            current_step = current_timestep - period.step_start
            current_w = period.assist_w_start + slope * current_step
            return current_w
        return 0


    def __check_schedule(self):
        prev_start, prev_end = -1, -1
        for period in self.injection_periods:
            self.__check_period_alone(period)
            if period.step_start < prev_start: raise ValueError("Injection period steps are not in order!")
            if prev_end > period.step_start: raise ValueError("Injection periods overlapping!")
            prev_start = period.step_start
            prev_end = period.step_end

    def __check_period_alone(self, period:InjectionPeriod):
        if min(period.step_start, period.step_end, period.assist_w_start, period.assist_w_end, period.noise_update_freq, 0) != 0: raise ValueError("Steps, weights or noise_std can't be negative for a InjectionPeriod!")
        if period.step_start >= period.step_end: raise ValueError("Period step start can't be >= than step end!")
        if max(period.assist_w_start,period.assist_w_end,1)>1: raise ValueError("Assist weights can't be bigger than 1")
