from injection.scheludes.injection_schedule import InjectionSchedule
from utils.enums import NoiseUpdateFreq
import enum
from utils.enums import AssitiveActors


SCH1 = InjectionSchedule(periods=[InjectionSchedule.InjectionPeriod(assistive_actor_type=AssitiveActors.mountaincar_basic,
                                                                    step_start=0, step_end=50_000, assist_w_start=0,
                                                                    assist_w_end=0, noise_std=0.3,
                                                                    min_w=0, max_w=0.5,
                                                                    noise_update_freq=NoiseUpdateFreq.every_episode,
                                                                    render=False)]
                         )

SCH2 = InjectionSchedule(periods=[InjectionSchedule.InjectionPeriod(assistive_actor_type=AssitiveActors.mountaincar_basic,
                                                                    step_start=0, step_end=20_000, assist_w_start=0,
                                                                    assist_w_end=0, noise_std=0.3,
                                                                    min_w=1, max_w=1,
                                                                    noise_update_freq=NoiseUpdateFreq.every_episode,
                                                                    render=False)])
                                  # InjectionSchedule.InjectionPeriod(assistive_actor_type=AssitiveActors.mountaincar_pretrained,
                                  #                                   step_start=75_000, step_end=100_000, assist_w_start=0,
                                  #                                   assist_w_end=0, noise_std=0.5,
                                  #                                   min_w=0, max_w=1,
                                  #                                   noise_update_freq=NoiseUpdateFreq.every_action,
                                  #                                   render=False),
                                  # InjectionSchedule.InjectionPeriod(assistive_actor_type=AssitiveActors.mountaincar_pretrained,
                                  #                                   step_start=150_000, step_end=200_000, assist_w_start=0,
                                  #                                   assist_w_end=0, noise_std=0.5,
                                  #                                   min_w=0, max_w=1,
                                  #                                   noise_update_freq=NoiseUpdateFreq.every_action,
                                  #                                   render=False)])

SCH3 = InjectionSchedule(periods=[InjectionSchedule.InjectionPeriod(assistive_actor_type=AssitiveActors.mountaincar_basic,
                                                                    step_start=0, step_end=250, assist_w_start=1,
                                                                    assist_w_end=1, noise_std=0,
                                                                    min_w=0, max_w=1,
                                                                    noise_update_freq=NoiseUpdateFreq.every_action,
                                                                    render=False),
                                  InjectionSchedule.InjectionPeriod(
                                      assistive_actor_type=AssitiveActors.mountaincar_basic,
                                      step_start=500, step_end=750, assist_w_start=1,
                                      assist_w_end=1, noise_std=0,
                                      min_w=0, max_w=1,
                                      noise_update_freq=NoiseUpdateFreq.every_action,
                                      render=False),

                                    InjectionSchedule.InjectionPeriod(
                                      assistive_actor_type=AssitiveActors.mountaincar_basic,
                                      step_start=1000, step_end=1250, assist_w_start=1,
                                      assist_w_end=1, noise_std=0,
                                      min_w=0, max_w=1,
                                      noise_update_freq=NoiseUpdateFreq.every_action,
                                      render=False),
                                  InjectionSchedule.InjectionPeriod(
                                      assistive_actor_type=AssitiveActors.mountaincar_basic,
                                      step_start=1750, step_end=2000, assist_w_start=1,
                                      assist_w_end=1, noise_std=0,
                                      min_w=0, max_w=1,
                                      noise_update_freq=NoiseUpdateFreq.every_action,
                                      render=False)
                                    ]
                         )


class InjectionSchedules(enum.Enum):
    sch1 = SCH1
    sch2 = SCH2
