from trainer import Trainer
from demo import Demonstrator
from hyperparams import Hyperparams

if __name__ == "__main__":
    seeds = [1,1001,3333] # 15,105,133,145,155 # ,5000,9449

    hyperparams_list = [Hyperparams(seed=seed, rollout_len=4096, max_episode_len=1000, lr=5e-3, min_lr=1e-4, advantage_calc="gae", gae_lambda=0.98, learn_std=False) for seed in seeds] # seed=seed, rollout_len=4096, max_episode_len=1000, lr=5e-3, min_lr=1e-4, advantage_calc="gae", gae_lambda=0.98, learn_std=False

    for env_name in ["LunarLanderContinuous-v2"]: # "Hopper-v4", HalfCheetah-v4", "InvertedPendulum-v4", "InvertedDoublePendulum-v4", "Swimmer-v4", "Walker2d-v4"
        trainer = Trainer(env_name=env_name, hyperparams_list=hyperparams_list, total_timesteps=1_000_000)
        trainer.train(plt_save_path=f"./models/{env_name}/performance.png")

    # demonstrator = Demonstrator(env_name="HalfCheetah-v4", render=True)
    # demonstrator.play()

