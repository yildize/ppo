from convenience.trainer import Trainer
from convenience.demonstrator import Demonstrator
from injection.scheludes.ready_schedules import InjectionSchedules
from utils.enums import AdvNormMethods
from utils.hyperparams import Hyperparams
from utils.utils import get_current_date_time_str

# This is the main script that a training or demo can be performed as follows:
if __name__ == "__main__":

    # Set your seeds
    seeds = [5, 55]#[1, 3333, 9449] # 3333 works fine?


    # hyperparams = []
    # for seed in seeds:
    #     for adv_norm_method in (AdvNormMethods.normalize, AdvNormMethods.not_normalize, AdvNormMethods.range_scale):
    #         for learn_std in (True, False):
    #             for batchify in (True, False):
    #                 hyperparams.append(Hyperparams(seed=seed, learn_std=learn_std, max_episode_len=500, normalize_obs=True, adv_norm_method=adv_norm_method, batchify=batchify))
    #

    # hyperparams = []
    # for seed in seeds:
    #         for learn_std in (False, True):
    #             for batchify in (True, False):
    #                 hyperparams.append(Hyperparams(seed=seed, learn_std=learn_std, max_episode_len=500, normalize_obs=True, adv_norm_method=AdvNormMethods.range_scale, batchify=batchify,
    #                                                injection_enabled=True))

    # # SCH1 injection increased to 50k steps 0.3 noise max 0.5w that's all!
    # hyperparams = []
    # for seed in seeds:
    #         for batchify in (True, False):
    #             for normalize_obs in (True, False):
    #                 for adv_norm_method in (AdvNormMethods.range_scale, AdvNormMethods.normalize, AdvNormMethods.not_normalize):
    #                     hyperparams.append(Hyperparams(seed=seed, learn_std=False, max_episode_len=500, normalize_obs=normalize_obs,
    #                                                    adv_norm_method=adv_norm_method, batchify=batchify, injection_enabled=True))


    # # Set hyperparameter list. Each environment will be trained with each hyperparams setup in the list.
    # # Default parameters are the best performer configuration for me.
    # hyperparams = [Hyperparams(seed=seed, learn_std=False, max_episode_len=500, normalize_obs=False, adv_norm_method=AdvNormMethods.range_scale,
    #                                 injection_enabled=True, injection_schedule=InjectionSchedules.sch2) for seed in seeds]


    hyperparams = []
    for seed in seeds:
        hyperparams.append(Hyperparams(seed=seed, learn_std=False, normalize_obs=False,
                                      adv_norm_method=AdvNormMethods.normalize, batchify=False, injection_enabled=True,
                                      injection_schedule=InjectionSchedules.sch3, max_episode_len=200))



    # Now provide the list of environment names to train.
    #MountainCarContinuous-v0
    for env_name in ["LunarLanderContinuous-v2"]:#["HalfCheetah-v4", "InvertedPendulum-v4", "InvertedDoublePendulum-v4", "Hopper-v4", "Reacher-v4", "Swimmer-v4", "Walker2d-v4", "LunarLanderContinuous-v2"]:
        # Trainer will train the given environment for each hyperparams setup provided in the list. For this case only the seed is changed.
        total_timesteps = 100_000
        trainer = Trainer(env_name=env_name, hyperparams_list=hyperparams, total_timesteps=total_timesteps)
        trainer.train(session_name=f"{env_name}_{total_timesteps}_{get_current_date_time_str()}") # performance plot result will be saved to the provided path.




    # #env_names = ["HalfCheetah-v4", "Hopper-v4", "InvertedDoublePendulum-v4", "InvertedPendulum-v4", "Reacher-v4", "Swimmer-v4", "Walker2d-v4"]
    # #Demonstrator uses the provided model (the model put inside the models/<env_name>/demo/) and runs num_episodes in a rendered way just for demo purposes.
    # demonstrator = Demonstrator(env_name="MountainCarContinuous-v0", num_episodes=1, render=True)
    # demonstrator.play()

