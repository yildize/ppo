import os
import gym
from utils.hyperparams import Hyperparams
from core.ppo import PPO
from utils.plotter import Plotter
from typing import List

from utils.render_wrapper import RenderWrapper
from utils.utils import PerformanceLogger, create_directory_if_not_exists
import json
from dataclasses import dataclass, asdict

class Trainer:
    """ This class is a utility class to run multiple experiments/trainings with the same environment and plot the results. Running multiple experiments
    will provide us a better confidence on the performance of our algorithm."""

    def __init__(self, env_name:str, hyperparams_list:List[Hyperparams], total_timesteps:int, plot:bool=True, plot_points:int=1000):
        self.total_timesteps = total_timesteps
        self.env_name = env_name
        self.hyperparams_list = hyperparams_list
        self.plot = plot
        self.plot_points = plot_points

    def train(self, session_name:str):
        """ Runs multiple experiments for each environment. Performance plot will be saved to the provided path."""
        logs = [] # will store performances for each seeded experiment.

        for i, hyperparams in enumerate(self.hyperparams_list):
            log = self.__train_model(hyperparams=hyperparams, session_name=session_name, session_train_index=i)
            logs.append(log)

        # Each train call is assumed to be on the same environment typically with different seeds.
        plt_save_path = os.path.join("logs", session_name, "performance.png")
        if self.plot: Plotter.plot(logs=logs, number_of_points=self.plot_points, title=f"{self.env_name}", save_path=plt_save_path)

    def __train_model(self, hyperparams:Hyperparams, session_name:str, session_train_index:int) -> PerformanceLogger:
        """ Runs a single experiment for the given environment, seed and timestep"""
        # Create the env
        env = RenderWrapper(env_name=self.env_name, normalize_obs=hyperparams.normalize_obs)

        # Create default hyperparams, I will me modifying the seed
        #hyperparams = Hyperparams()
        #hyperparams.seed = seed
        learner = PPO(env, hyperparams=hyperparams)

        # Let's start learning
        learner.learn(total_timesteps=self.total_timesteps)

        # Create the session folder to store the training results
        path_to_session_folder = os.path.join("logs", session_name)
        create_directory_if_not_exists(path_to_session_folder)

        # Now save the model using the last reward and the env name. Note that we could have save the best model as well.
        last_rollout_reward = learner.perf_logger.avg_eps_rews[-1]
        base_model_save_path = os.path.join(path_to_session_folder, f"{session_train_index}_rew{last_rollout_reward:.2f}")
        # Save actor model
        learner.actor_critic_networks.save(path=base_model_save_path, only_actor=True)
        # Save obs_rms
        if env.normalize_obs: env.save_obs_rms(filename=os.path.join(path_to_session_folder, f"obs_rms_{session_train_index}"))

        # Finally save the hyperparams
        hyperparams.injection_schedule = hyperparams.injection_schedule.name # It is not used in demo, so just convert object to str, otherwise problems occur.
        hyperparams_dict = asdict(hyperparams)
        with open(os.path.join(path_to_session_folder, f"hyperparams_{session_train_index}.json"), 'w') as f:
            json.dump(hyperparams_dict, f, indent=4)

        # Return the recorded performance metrics
        return learner.perf_logger

