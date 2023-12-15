import numpy as np
import torch
import gym
import json

from typing import Union

from utils.render_wrapper import RenderWrapper
from utils.utils import find_first_file_in_directory, load_with_pickle
from core.networks import MLP
import time
import warnings


class Demonstrator:

    env: Union[gym.Env, RenderWrapper]
    """ This class laods a trained actor and play the game deterministically for demo purposes."""
    def __init__(self, env_name:str, num_episodes:int=5, max_episode_length:int=1000, render=True, step_delay=0.01,):
        self.env_name = env_name

        self.render = render
        self.step_delay = step_delay
        self.hyperparams_dict = self.__load_hyperparams_dict()
        self.normalize_obs = self.hyperparams_dict["normalize_obs"]

        # Create the env
        self.env = RenderWrapper(env_name=env_name, normalize_obs=self.normalize_obs)
        if render: self.env.activate_rendering()
        if self.normalize_obs: self.env.freeze_obs_rmss()  # I don't want to update obs_rmsses during demo, I used method injection for that which is not the best practice to be honest.

        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length

        self.policy = self.__create_empty_policy_network()
        self.__load_demo_actor()
        self.__load_obs_rms()

    def play(self):
        """ Plays for num_episodes episodes and prints each episode performance along with the overall average perf."""
        eps_lens, eps_rewards = [], []
        for eps in range(self.num_episodes):
            eps_len, eps_rew = self._play_an_episode()
            eps_lens.append(eps_len), eps_rewards.append(eps_rew)
            print(f"Episode {eps+1}/{self.num_episodes} -> Length: {eps_len} - Reward: {eps_rew}")
        print(f"Average episode length: {np.mean(eps_lens)} - Average episode rewards: {np.mean(eps_rewards)}")

    def _play_an_episode(self):
        """ This method plays a single episode with the provided env and agent."""
        s, _  = self.env.reset()
        episode_rewards = []
        for t in range(self.max_episode_length):
            if self.render: time.sleep(self.step_delay)
            action = self._decide_action(state=s)
            print("action:", action)
            s, r, done, truncated, _ = self.env.step(action=action)
            episode_rewards.append(r)
            if done or truncated: break

        eps_len = t+1
        eps_reward = sum(episode_rewards)

        return eps_len, eps_reward

    def _decide_action(self, state:np.ndarray) -> np.ndarray:
        """ Just selects the action. Note that since it is a demo mode, we directly use the
        mean for the gaussian distribution, thus there is no exploration."""
        res = self.policy(state)
        if isinstance(res, tuple): mean, std = res
        else: mean = res
        return mean.detach().numpy()

    def __load_hyperparams_dict(self):
        # First read the hyperparams from the demo folder to reconstruct the actor:
        hyperparams_path = find_first_file_in_directory(directory_path=f"./models/{self.env_name}/demo", containing="hyperparams")
        if hyperparams_path is None: raise FileNotFoundError(f"Please place a hyperparams file inside the ./models/{self.env_name}/demo so that, Demonstrator"
                                                             f"can reconstruct the same actor.")
        with open(hyperparams_path, 'r') as f:
            hyperparams_dict = json.load(f)
        return hyperparams_dict

    def __create_empty_policy_network(self, ) -> torch.nn.Module:
        # Construct the policy/actor network same with the saved model so that we can successfully load the weights.
        policy:torch.nn.Module = MLP(input_dim=self.env.observation_space.shape[0], output_dim=self.env.action_space.shape[0],
                                     hidden_dim=self.hyperparams_dict["hidden_dim"], num_hidden_layers=self.hyperparams_dict["num_hidden_layers_actor"],
                                     learn_std=self.hyperparams_dict["learn_std"], tanh_acts=self.hyperparams_dict["tanh_acts"])
        return policy


    def __load_demo_actor(self):
        try:
            model_path = find_first_file_in_directory(directory_path=f"./models/{self.env_name}/demo", containing="actor")
            if model_path is None: raise FileNotFoundError
        except:
            raise Exception(f"Please be sure you have placed the correct actor model inside ./models/{self.env_name}/demo. Also make sure"
                            f"the trained actor network is the same as the demo policy network.")

        # Load in the actor model saved by the PPO algorithm
        self.policy.load_state_dict(torch.load(model_path))

    def __load_obs_rms(self):
        # Load the obs_rms as well if it is put inside the demo
        obs_rms_path = find_first_file_in_directory(directory_path=f"./models/{self.env_name}/demo", containing="obs_rms")
        if not self.normalize_obs and obs_rms_path: warnings.warn("You haven't set norm_obs for demonstrator, but I have found and obs_rms inside demo. Do you want"
                                                                  "a normalized demonstration? Then set 'normalize_obs'=True ")
        if self.normalize_obs:
            if obs_rms_path:
                obs_rms = load_with_pickle(file_path=obs_rms_path)
                self.env.set_obs_rms(obs_rms)
            else:
                raise Exception(f"You have wanted a normalized obs. demo, but I couldn't find a obs_rms file inside ./models/{self.env_name}/demo.")


