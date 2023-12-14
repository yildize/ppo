

import os
import torch
from torch.distributions import MultivariateNormal
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np



class MultivariateGaussianDist:
    """ I will use this class to represent my action distribution. The actor network will provide me
    the mean vector for my continuous action space. But I need a probabilistic policy.

    It allows us to sample actions in a way that is more likely to explore around the mean,
    which the actor network predicts as the most probable good action.

    Note that here I will be using a fixed covariance matrix with fixed std_dev. This can be learned by a network
    as well if the environment is dynamic.

    Here is a short youtube video quikcly explaining multivariate normal distribution and the effect of covariance matrix on it.
    #         # https://www.youtube.com/watch?v=azrTdjrA2bU
    """
    def __init__(self, num_actions:int, std_dev_start:float, std_dev_end:float):
        assert (std_dev_start>=std_dev_end)
        self.num_actions = num_actions
        self.std_dev = self.std_dev_start = std_dev_start
        self.std_dev_end = std_dev_end
        self.construct_cov_matrix()
        self.total_timesteps = None

    def construct_cov_matrix(self):
        """ This method will simply construct a covariance matrix using a fixed std_dev value for each action dimension."""
        self.cov_var = torch.full(size=(self.num_actions,), fill_value=self.std_dev)
        self.cov_mat = torch.diag(self.cov_var)

    def update_cov_matrix(self, timestep):
        """ This method is and optional method that is used to update/reduce the std_dev by time to reduce the exploration
        and increase the exploitation by time."""
        fraction = timestep/self.total_timesteps
        self.std_dev = self.std_dev_start - fraction*(self.std_dev_start-self.std_dev_end)
        self.construct_cov_matrix()

    def gaussian_action_dist(self, mean: torch.Tensor, std:torch.Tensor) -> MultivariateNormal:
        """ This function will return a gaussian action distribution using the provided mean vector and the
        predefined covariance matrix."""
        if std is None: # Non learnable std
            cov_mat = self.cov_mat
        elif std.dim() == 1: # Single action case
            cov_mat = torch.diag(std ** 2)
        else:
            assert(std.dim()==2)
            batch_size, action_dim = std.shape
            expanded_std = std.unsqueeze(2).expand(batch_size, action_dim, action_dim)
            identity = torch.eye(action_dim).unsqueeze(0).expand(batch_size, -1, -1)
            cov_mat = identity * expanded_std.pow(2)

        return MultivariateNormal(mean, cov_mat)


class PerformanceLogger:
    """ This is a simple class to store rollout performances (length and total rewards) along with corresponding timesteps
    those logs will be used to obtain performance plots."""
    def __init__(self):
        self.avg_eps_lens:List[float] = []  # avg episode lengths for each rollout
        self.avg_eps_rews:List[float] = []  # avg episode rewards for each rollout.
        self.timesteps: List[int] = [] # value of timestep for each recording.

    def add(self, avg_episodic_lengths: float, avg_episodic_rewards: float, timestep:int):
        """ Adds a new rollout performance data"""
        self.avg_eps_lens.append(avg_episodic_lengths)
        self.avg_eps_rews.append(avg_episodic_rewards)
        self.timesteps.append(timestep)

    def plot(self):
        """ Optional method to plot the current data."""
        plt.plot(self.timesteps, self.avg_eps_rews)
        plt.xlabel("Timesteps")
        plt.ylabel("Average Rollout Reward")
        plt.show()


def batchify(batchsize:int, states_tensor:torch.Tensor, actions_tensor:torch.Tensor, initial_log_probs_tensor:torch.Tensor, A:torch.Tensor, monte_carlo_qas:torch.Tensor):
    """ Utility function used to yield batches of given tensors with given batchsize. Note that I will be selecting batches with replacement to prevent dimension errors."""
    rollout_len = states_tensor.shape[0]
    rollout_indices = np.arange(rollout_len)
    np.random.shuffle(rollout_indices)

    for _ in np.arange(0, rollout_len, batchsize):
        batch_indices = np.random.choice(rollout_indices, batchsize, replace=False)
        yield states_tensor[batch_indices], actions_tensor[batch_indices], initial_log_probs_tensor[batch_indices], A[batch_indices], monte_carlo_qas[batch_indices]


def create_directory_if_not_exists(directory_path):
    """ Utility function to create the directory if not exists."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def find_first_file_in_directory(directory_path):
    """Finds the first file in the given directory path."""
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Return the full path of the first file found
            return os.path.join(root, file)
    return None