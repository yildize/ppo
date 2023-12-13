import torch
from typing import Tuple, List
from utils import MultivariateGaussianDist
import numpy as np

class RolloutComputer:
    """ Objective of this class is to abstract important calculations done on the rollout"""
    def __init__(self):
        ...
    @staticmethod
    def convert_list_to_tensor(given_list:List):
        """Converts list to tensor"""
        return torch.tensor(np.array(given_list), dtype=torch.float32)

    @staticmethod
    def estimated_qas(next_states:List[np.ndarray], rewards:List[float], dones:List[bool], truncateds:List[bool], critic:torch.nn.Module,  discount_factor:float) -> torch.Tensor:
        """ This method will calculate the estimate state action value Q(s,a)'s for each collected state of the rollout"""

        # Check if we have states to calculate the estimated Q(a,s)'s for.
        if not len(next_states): raise ValueError("You are trying to calculate estimated Q(a,s) on an empty rollout_buffer.states")

        # Initialize the list for estimates Q(s,a) values.
        estimated_qas = np.zeros_like(rewards)  # some sources call it rewards/returns to go as well. To me it seems like Q(s,a) estimations.

        # We start from the end of the rollout and accumulate the reward
        for i in reversed(range(len(rewards))):
            # If the episode is done or truncated at the next step, we reset the accumulated reward
            # Note: It is important to reset at the next step, not the current one, since even if done is True, the current reward is still part of the episode
            if dones[i]:
                G = 0
            elif truncateds[i]:
                # Must be truncated or maybe max_episode length is reached again it is a kind of truncation
                # In this case, the next state value is not zero, thus let's estimate it using the critic
                G = critic(next_states[i]).detach().numpy()[0]

            G = rewards[i] + (discount_factor * G)
            estimated_qas[i] = G

        # Return estimated Q(s,a) values for each rollout state as a tensor
        return torch.tensor(estimated_qas, dtype=torch.float32)
    @staticmethod
    def value_estimates(states:torch.Tensor, critic:torch.nn.Module) -> torch.Tensor:
        """ Simply calculates value estimates for the given state utilizing the critic."""
        # Get the value predictions for each state in the buffer:
        V = critic(states).squeeze()
        return V

    @staticmethod
    def curr_log_probs(states:torch.Tensor, actions:torch.Tensor, actor:torch.nn.Module, multivariate_gaussian_dist:MultivariateGaussianDist)  -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate current log probabilities for the rollout actions."""
        # First get the expected actions for each state in the buffer
        ress = actor(states)
        if isinstance(ress, tuple): means, stds = ress
        else: means,stds = ress, None

        # Since we'll be feeding a batch of means, we should expect a batch of action distributions
        act_dists = multivariate_gaussian_dist.gaussian_action_dist(mean=means, std=stds)
        # Now we'll calculate the log_probs for each action taken in the rollout
        log_probs = act_dists.log_prob(actions)
        return log_probs, act_dists.entropy()

    @staticmethod
    def advantage_estimates(estimated_qas:torch.Tensor, estimated_values:torch.Tensor, normalized:bool=True) -> torch.Tensor:
        """ Calculate the basic advantage estimates using rewards to go and estimated values."""
        A = estimated_qas - estimated_values
        if normalized: A = (A - A.mean()) / (A.std() + 1e-10) # # 1e-8 is just use to avoid possible divide by 0 condition.
        return A

    @staticmethod
    def gae(rewards:List[float], values:List[float], last_state_val:float, dones:List[bool], gamma:float, gae_lambda:float):
        """ Generalized Advantage Estimates calculation. It provides a tradeoff between variance and bias between estimates
        through the gae_lambda parameter. """
        rollout_len = len(rewards)
        advantages = torch.zeros(rollout_len)
        gae = 0
        # Pre-set the next_val for the last timestep if not done
        next_val = last_state_val if not dones[-1] else 0
        for t in reversed(range(rollout_len)):
            # Use the pre-set next_val for the last timestep
            if t == rollout_len - 1:
                delta = rewards[t] + gamma * next_val - values[t]
            else:
                next_val = 0 if dones[t] else values[t + 1]
                delta = rewards[t] + gamma * next_val - values[t]
            gae = delta + gamma * gae_lambda * gae * (1 - dones[t])  # Reset to 0 if the state is terminal
            advantages[t] = gae
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        return advantages

    @staticmethod
    def new_policy_is_diverged(initial_log_probs_tensor, importance_sampling_ratios, curr_log_probs, max_kl_divergence):
        """ This method approximates KL divergence and tells if the current policy is diverged too much after the updates."""
        approximate_kl_divergence = ((importance_sampling_ratios - 1) - (curr_log_probs - initial_log_probs_tensor)).mean()
        return approximate_kl_divergence > max_kl_divergence


    # @staticmethod
    # def gae3(rewards, values, last_state_val, dones, gamma, gae_lambda):
    #     rollout_len = len(rewards)
    #     gae = 0
    #     advantages = torch.zeros(rollout_len)
    #
    #     for t in reversed(range(rollout_len)):
    #         if t == rollout_len-1:
    #             next_val = last_state_val if not dones[t] else 0
    #         else:
    #             next_val = 0 if dones[t] else values[t + 1]
    #
    #         td = rewards[t] + gamma * next_val - values[t]
    #         gae = td + gamma * gae_lambda * gae * (1 - dones[t])
    #         advantages[t] = gae
    #
    #     #advantages *= (1-gae_lambda)
    #     #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
    #     return advantages
    #
    # @staticmethod
    # def gae1(rewards, values, last_state_val, dones, gamma, gae_lambda):
    #     rollout_len = len(rewards)
    #     advantages = torch.zeros(rollout_len)
    #     gae = 0
    #     for t in reversed(range(rollout_len)):
    #         next_val = 0 if dones[t] else values[t + 1]
    #         if t == rollout_len - 1 and not dones[t]:
    #             next_val = last_state_val
    #         delta = rewards[t] + gamma * next_val - values[t]
    #         gae = delta + gamma * gae_lambda * gae * (1 - dones[t])  # Reset to 0 if the state is terminal
    #         advantages[t] = gae
    #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
    #     return advantages
    #
    # @staticmethod
    # def gae2(rewards, values, dones, gamma, gae_lambda):
    #     gae = 0
    #     advantages = []
    #
    #     for t in reversed(range(len(rewards))):
    #         if t == len(rewards) - 1:
    #             next_non_terminal = 1.0 - dones[t]
    #             next_values = 0
    #         else:
    #             next_non_terminal = 1.0 - dones[t + 1]
    #             next_values = values[t + 1]
    #
    #         delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
    #         gae = delta + gamma * gae_lambda * next_non_terminal * gae
    #         advantages.insert(0, gae + values[t])  # Inserting at the beginning of the list
    #
    #     return torch.tensor(advantages, dtype=torch.float32)



