import gym
from hyperparams import Hyperparams
from ppo import PPO
from plotter import Plotter
from typing import List
from utils import PerformanceLogger, create_directory_if_not_exists


class Trainer:
    """ This class is a utility class to run multiple experiments/trainings with the same environment and plot the results. Running multiple experiments
    will provide us a better confidence on the performance of our algorithm."""

    def __init__(self, env_name:str, hyperparams_list:List[Hyperparams], total_timesteps:int, plot:bool=True, plot_points:int=1000):
        self.total_timesteps = total_timesteps
        self.env_name = env_name
        self.hyperparams_list = hyperparams_list
        self.plot = plot
        self.plot_points = plot_points

    def train(self, plt_save_path:str):
        """ Runs multiple experiments for each environment. Performance plot will be saved to the provided path."""
        logs = [] # will store performances for each seeded experiment.
        for hyperparams in self.hyperparams_list:
            log = self.__train_model(hyperparams=hyperparams)
            logs.append(log)

        if self.plot: Plotter.plot(logs=logs, number_of_points=self.plot_points, title=f"{self.env_name}", save_path=plt_save_path)

    def __train_model(self, hyperparams:Hyperparams) -> PerformanceLogger:
        """ Runs a single experiment for the given environment, seed and timestep"""
        # Create the env
        env = gym.make(self.env_name)

        # Create default hyperparams, I will me modifying the seed
        #hyperparams = Hyperparams()
        #hyperparams.seed = seed
        learner = PPO(env, hyperparams=hyperparams)

        # Let's start learning
        learner.learn(total_timesteps=self.total_timesteps)

        # Now save the model using the last reward and the env name. Note that we could have save the best model as well.
        last_rollout_reward = learner.perf_logger.avg_eps_rews[-1]
        base_save_path = f"./models/{self.env_name}/{last_rollout_reward:.2f}"
        create_directory_if_not_exists(f"./models/{self.env_name}")
        learner.actor_critic_networks.save(path=base_save_path, only_actor=True)

        # Return the recorded performance metrics
        return learner.perf_logger

