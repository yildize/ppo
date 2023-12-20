import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from utils.utils import MultivariateGaussianDist
from typing import Optional, Tuple

class MLP(nn.Module):
    """ This is a structure for a simple MLP network which will be used for both actor and critic. Obviously
    we could have used more complex networks, it is relatively easy to do it through torch. But the main point
    of PPO is not the network architectures but more like a way of learning their parameters."""

    def __init__(self, input_dim:int, output_dim:int, hidden_dim:int, learn_std=False, tanh_acts=False, num_hidden_layers=2):
        super().__init__()
        self.activation_fn = F.tanh if tanh_acts else F.relu
        self.learn_std = learn_std

        # Define the MLP layers:
        self.linear_layer1 = nn.Linear(input_dim, hidden_dim)
        # I will provide hidden layers as a list to make it more flexible
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.linear_layer3 = nn.Linear(hidden_dim, output_dim)

        # Define learnable std params if specified
        if learn_std: self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, obs):
        """ Define the forward propagation logic. Torch will handle the gradient calculation. """
        # Conversion from numpy array to torch tensor just in case:
        #obs = torch.tensor(obs, dtype=torch.float32) if isinstance(obs, np.ndarray) else obs
        obs = torch.as_tensor(obs, dtype=torch.float32)

        # Now let's calculate the forward passes:
        a = self.activation_fn(self.linear_layer1(obs))  # a = F.tanh(self.linear_layer1(obs)) if self.tanh_acts else F.relu(self.linear_layer1(obs))
        for hl in self.hidden_layers: a = self.activation_fn(hl(a))
        logits = self.linear_layer3(a)

        # If stds are learnable output stds along with the logits
        if self.learn_std:
            return logits, torch.exp(self.log_std)  # returns mean and std

        return logits


class ActorCriticNetworks:
    """ This is a utility container collecting all the network instances and their optimizers to provide
    ease of usage. Our PPO learner instance will utilize an actor critic instance."""

    def __init__(self, num_observations:int, num_actions:int, hidden_dim:int, num_hidden_layers_actor: int, num_hidden_layers_critic:int,  multivariate_gauss_dist:MultivariateGaussianDist, learn_std:bool, tanh_acts:bool,  lr=3e-4):
        # Construct the actor and critic networks according to environment observation and action spaces:
        self.actor = MLP(input_dim=num_observations, output_dim=num_actions, hidden_dim=hidden_dim, learn_std=learn_std, tanh_acts=tanh_acts, num_hidden_layers=num_hidden_layers_actor)
        self.critic = MLP(input_dim=num_observations, output_dim=1, hidden_dim=hidden_dim, tanh_acts=tanh_acts, num_hidden_layers=num_hidden_layers_critic)

        # Set lr we can use it later to update it.
        self.lr = self.initial_lr = lr
        # I will be using Adam optimizer as the paper suggests
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # I will be using a multivariate gaussian distribution to introduce exploration to the agent.
        self.multivariate_gauss_dist = multivariate_gauss_dist
        self.total_timesteps = None

    def sample_an_action(self, state:np.ndarray) -> Tuple[np.ndarray, float]:
        """ This method will output an action and its log probability for the current state using the actor network
        and the multivariate gaussian distribution"""
        # I don't want to compute gradients for this operation.
        with (torch.no_grad()):
            # Get current actor action probs they will act as the mean probs
            res = self.actor(state) # this will be the mean probs vector of shape (num_actions,)
            if isinstance(res, tuple): mean, std = res
            else: mean, std = res, None
            # [num_actions]

            # Create a multivariate normal distribution with mean and covariance matrix
            action_dist = self.multivariate_gauss_dist.gaussian_action_dist(mean=mean, std=std)

            # Sample an action from the distribution:
            action = action_dist.sample() # so this will be continous action

            # [num_actions]
            # Also calculate the log prob
            log_prob_a = action_dist.log_prob(action)

        return action.numpy(), log_prob_a.item()

    def take_a_gradient_step(self, actor_loss:torch.Tensor, critic_loss:torch.Tensor, clip_grad:bool, max_grad:Optional[float]=None):
        """ Here I will be first calculating the gradients on the learnable parameters then take a gradient step
        using the Adam optimizers. """

        # Take a gradient step for actor:
        self.actor_optim.zero_grad()  # zeroize gradients first because it is accumulated.
        actor_loss.backward()  # this calculates the gradients through chain rule.
        if clip_grad and max_grad is not None: nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad)
        self.actor_optim.step()  # takes a gradient step using the calculated gradients.

        # Take a gradient step for the critic
        self.critic_optim.zero_grad()  # zeroize gradients first because it is accumulated.
        critic_loss.backward()  # this calculates the gradients through chain rule.
        if clip_grad and max_grad is not None: nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad)
        self.critic_optim.step()  # takes a gradient step using the calculated gradients.

    def reduce_learning_rate(self, curr_timestep: int, min_lr: float):
        """ This method is used to reduce the learning rate linearly by current timestep"""
        frac_complete = curr_timestep/self.total_timesteps
        new_lr = (1 - frac_complete) * self.initial_lr + frac_complete * min_lr
        self.lr = max(new_lr, min_lr)
        self.actor_optim.param_groups[0]["lr"] = self.lr
        self.critic_optim.param_groups[0]["lr"] = self.lr

    def save(self, path:str="./", only_actor=False):
        """ This method is used to save the actor and critic (optionally)  network to the provided path."""
        actor_path = path+"_ppo_actor"
        torch.save(self.actor.state_dict(), actor_path)
        if not only_actor:
            critic_path = path + "_ppo_critic"
            torch.save(self.critic.state_dict(), critic_path)