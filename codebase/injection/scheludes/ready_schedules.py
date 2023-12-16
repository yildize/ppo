from injection.scheludes.injection_schedule import InjectionSchedule
from utils.enums import NoiseUpdateFreq
import enum
from utils.enums import AssitiveActors


SCH1 = InjectionSchedule(periods=[InjectionSchedule.InjectionPeriod(assistive_actor_type=AssitiveActors.mountaincar_basic,
                                                                    step_start=0, step_end=50_000, assist_w_start=0.3,
                                                                    assist_w_end=0.0, noise_std=0.01,
                                                                    min_w=0, max_w=0.35,
                                                                    noise_update_freq=NoiseUpdateFreq.every_episode,
                                                                    render=False)])

SCH2 = InjectionSchedule(periods=[InjectionSchedule.InjectionPeriod(assistive_actor_type=AssitiveActors.mountaincar_basic,
                                                                    step_start=0, step_end=100_000, assist_w_start=0.1,
                                                                    assist_w_end=0.0, noise_std=0.25,
                                                                    min_w=0, max_w=0.3,
                                                                    noise_update_freq=NoiseUpdateFreq.every_episode,
                                                                    render=False)])


class InjectionSchedules(enum.Enum):
    sch1 = SCH1
    sch2 = SCH2
