from injection.scheludes.injection_schedule import InjectionSchedule
from utils.enums import NoiseUpdateFreq
import enum
from utils.enums import AssitiveActors


SCH1 = InjectionSchedule(periods=[InjectionSchedule.InjectionPeriod(assistive_actor_type=AssitiveActors.mountaincar_basic,
                                                                    step_start=0, step_end=10_000, assist_w_start=0.3,
                                                                    assist_w_end=0.0, noise_std=0.15,
                                                                    min_w=0, max_w=0.3,
                                                                    noise_update_freq=NoiseUpdateFreq.every_episode,
                                                                    render=True)])

SCH2 = InjectionSchedule(periods=[InjectionSchedule.InjectionPeriod(assistive_actor_type=AssitiveActors.mountaincar_basic,
                                                                    step_start=0, step_end=15_000, assist_w_start=0.3,
                                                                    assist_w_end=0.0, noise_std=0,
                                                                    min_w=0, max_w=0.3,
                                                                    noise_update_freq=NoiseUpdateFreq.every_action,
                                                                    render=False)])


class InjectionSchedules(enum.Enum):
    sch1 = SCH1
    sch2 = SCH2
