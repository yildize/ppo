from convenience.trainer import Trainer
from convenience.demonstrator import Demonstrator
from injection.scheludes.ready_schedules import InjectionSchedules
from utils.enums import AdvNormMethods
from utils.hyperparams import Hyperparams
from utils.utils import get_current_date_time_str

mode = ["train","demo"][0]
env_name = ["MountainCarContinuous-v0", "LunarLanderContinuous-v2"][0]

# This is the main script that a training or demo can be performed as follows:
if __name__ == "__main__":

    if mode == "train":
        # Set your seeds
        seeds = [5, 55, 555]

        # Total training timesteps:
        total_timesteps = 300_000

        # Setup the hyperparameters for the training:
        # For the following hyperparams there will be 36 different trainings 3 seeds * 2 norm options * 2 batch options * 3 adv_norm_options.
        # Also injection is enabled and pre-determined schedule sch1 is utilized as the schedule.
        # This schedule injects noisy weighted actions for the first 50k trainings.
        hyperparams = []
        for seed in seeds:
            for normalize_obs in (True, False):
                for batchify in (True, False):
                    for adv_norm_method in AdvNormMethods:
                        hyperparams.append(Hyperparams(injection_enabled=True, injection_schedule=InjectionSchedules.sch1,
                                                       seed=seed, learn_std=False, normalize_obs=normalize_obs,
                                                       adv_norm_method=adv_norm_method, batchify=batchify,
                                                       max_episode_len=500))

        # Trainer will train the given environment for each hyperparams setup provided in the list. For this case only the seed is changed.
        trainer = Trainer(env_name=env_name, hyperparams_list=hyperparams, total_timesteps=total_timesteps)
        trainer.train(session_name=f"{env_name}_{total_timesteps}_{get_current_date_time_str()}") # performance plot result will be saved to the provided path.
        # Training results will be saved under logs folder!

    else:
        # Demonstrator uses the provided model (the model put inside the models/<env_name>/demo/) and runs num_episodes in a rendered way just for demo purposes.
        # Also do not forget to provide hyperparams config file and the obs_rms if observation is normalized (you can obtain them from logs)
        demonstrator = Demonstrator(env_name=env_name, num_episodes=5, render=True)
        demonstrator.play()

