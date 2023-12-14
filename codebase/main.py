from convenience.trainer import Trainer
from convenience.demo import Demonstrator
from utils.hyperparams import Hyperparams

# This is the main script that a training or demo can be performed as follows:
if __name__ == "__main__":


    # Set your seeds
    seeds = [1, 3333, 9449]

    # Set hyperparameter list. Each environment will be trained with each hyperparams setup in the list.
    # Default parameters are the best performer configuration for me.
    hyperparams_list = [Hyperparams(seed=seed, learn_std=True, lr=7e-4, min_lr=7e-5) for seed in seeds]

    # Now provide the list of environment names to train.
    for env_name in ["MountainCarContinuous-v0"]:#["HalfCheetah-v4", "InvertedPendulum-v4", "InvertedDoublePendulum-v4", "Hopper-v4", "Reacher-v4", "Swimmer-v4", "Walker2d-v4", "LunarLanderContinuous-v2"]:
        # Trainer will train the given environment for each hyperparams setup provided in the list. For this case only the seed is changed.
        trainer = Trainer(env_name=env_name, hyperparams_list=hyperparams_list, total_timesteps=50_000_000)
        trainer.train(plt_save_path=f"./models/{env_name}/performance.png") # performance plot result will be saved to the provided path.


    # env_names = ["HalfCheetah-v4", "Hopper-v4", "InvertedDoublePendulum-v4", "InvertedPendulum-v4", "Reacher-v4", "Swimmer-v4", "Walker2d-v4"]
    # # Demonstrator uses the provided model (the model put inside the models/<env_name>/demo/) and runs num_episodes in a rendered way just for demo purposes.
    # demonstrator = Demonstrator(env_name="LunarLanderContinuous-v2", num_episodes=5, render=True)
    # demonstrator.play()

