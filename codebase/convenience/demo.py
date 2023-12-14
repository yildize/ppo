import numpy as np
import torch
import gym

from utils.utils import find_first_file_in_directory
from core.networks import MLP
import time

class Demonstrator:
    """ This class laods a trained actor and play the game deterministically for demo purposes."""
    def __init__(self, env_name:str, num_episodes:int=5, max_episode_length:int=1000, render=True, step_delay=0.01):
        self.render = render
        self.step_delay = step_delay
        # Create the env
        self.env = gym.make(env_name, render_mode="human") if render else gym.make(env_name)
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length

        self.policy:torch.nn.Module = MLP(input_dim=self.env.observation_space.shape[0], output_dim=self.env.action_space.shape[0], hidden_dim=64)

        try:
            model_path = find_first_file_in_directory(directory_path=f"./models/{env_name}/demo")
            if model_path is None: raise FileNotFoundError
        except:
            raise Exception(f"Please be sure you have placed the correct actor model inside ./models/{env_name}/demo. Also make sure"
                            f"the trained actor network is the same as the demo policy network.")

        # Load in the actor model saved by the PPO algorithm
        self.policy.load_state_dict(torch.load(model_path))

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
            s, r, done, truncated, _ = self.env.step(action=action)
            episode_rewards.append(r)
            if done or truncated: break

        eps_len = t+1
        eps_reward = sum(episode_rewards)

        return eps_len, eps_reward

    def _decide_action(self, state:np.ndarray) -> np.ndarray:
        """ Just selects the action. Note that since it is a demo mode, we directly use the
        mean for the gaussian distribution, thus there is no exploration."""
        mean = self.policy(state)
        return mean.detach().numpy()

