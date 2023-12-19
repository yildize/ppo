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

SCH2 = InjectionSchedule(periods=[InjectionSchedule.InjectionPeriod(assistive_actor_type=AssitiveActors.lunar_lander_joystick,
                                                                    step_start=0, step_end=300_000, assist_w_start=1,
                                                                    assist_w_end=0, noise_std=0.25,
                                                                    min_w=0, max_w=1,
                                                                    noise_update_freq=NoiseUpdateFreq.every_episode,
                                                                    render=True)])
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



periods = []
for rng in ((0,1000), (2000,3000),
            (5000,6000), (10000,11000),
            (12_000, 13_000), (14_000,15_000),
            ):
    for i in range(10):
        periods.append(InjectionSchedule.InjectionPeriod(assistive_actor_type=AssitiveActors.lunar_lander_joystick,
                                                         step_start=rng[0], step_end=rng[1], assist_w_start=1,
                                                         assist_w_end=1, noise_std=0,
                                                         min_w=1, max_w=1,
                                                         noise_update_freq=NoiseUpdateFreq.every_episode,
                                                         render=True))

    periods.append(InjectionSchedule.InjectionPeriod(assistive_actor_type=AssitiveActors.lunar_lander_joystick,
                                                     step_start=35_000, step_end=50_000, assist_w_start=1,
                                                     assist_w_end=0, noise_std=0.3,
                                                     min_w=0, max_w=1,
                                                     noise_update_freq=NoiseUpdateFreq.every_episode,
                                                     render=True))

SCH3 = InjectionSchedule(periods=periods)


class InjectionSchedules(enum.Enum):
    sch1 = SCH1
    sch2 = SCH2
    sch3 = SCH3
