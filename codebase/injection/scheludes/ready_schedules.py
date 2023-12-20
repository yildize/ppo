from injection.scheludes.injection_schedule import InjectionSchedule
from utils.enums import NoiseUpdateFreq
import enum
from utils.enums import AssitiveActors


# THIS SCRIPTS ENCAPSULATES PREDETERMINED INJECTION SCHEDULES TO PROVIDE EASE OF USE
# MORE SPECIFICALLY TO ALLOW SELECTION OF DIFFERENT SCHEDULES THROUGH ONE CHANGE ON THE HYPERPARAMS.

SCH1 = InjectionSchedule(periods=[InjectionSchedule.InjectionPeriod(assistive_actor_type=AssitiveActors.mountaincar_basic,
                                                                    step_start=0, step_end=50_000, assist_w_start=0,
                                                                    assist_w_end=0, noise_std=0.3,
                                                                    min_w=0, max_w=0.5,
                                                                    noise_update_freq=NoiseUpdateFreq.every_episode,
                                                                    render=False)])

SCH2 = InjectionSchedule(periods=[InjectionSchedule.InjectionPeriod(assistive_actor_type=AssitiveActors.mountaincar_joystick,
                                                                    step_start=0, step_end=50_000, assist_w_start=0,
                                                                    assist_w_end=0, noise_std=0.3,
                                                                    min_w=1, max_w=1,
                                                                    noise_update_freq=NoiseUpdateFreq.every_episode,
                                                                    render=False)])



class InjectionSchedules(enum.Enum):
    sch1 = SCH1
    sch2 = SCH2