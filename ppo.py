import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.nn as nn
import random

from utils import MultivariateGaussianDist, PerformanceLogger
from networks import ActorCriticNetworks
from rollout_buffer import RolloutBuffer
from rollout_computer import RolloutComputer
from typing import Tuple
from utils import batchify
from hyperparams import Hyperparams


class PPO:
    """ This is the class in which I have implemented to reproduce the results of original PPO paper."""

    def __init__(self, env: gym.Env, hyperparams:Hyperparams = Hyperparams()):
        # The environment we are trying to solve.
        self.env = env
        self.hyperparams = hyperparams
        # Seed everything
        self.seed_session(seed=self.hyperparams.seed)

        # I will be working with continuous control problems (main focus on the paper.)
        if type(env.observation_space) != gym.spaces.Box or type(env.action_space) != gym.spaces.Box:
            raise TypeError("This implementation expects a continous state and action spaces.")
        # Get the number of observations and number of actions using the environment
        self.num_observations, self.num_actions = env.observation_space.shape[0], env.action_space.shape[0]
        # I will be using a multivariate gaussian distribution to introduce exploration to the agent.
        self.multivariate_gauss_dist = MultivariateGaussianDist(num_actions=self.num_actions, std_dev_start=self.hyperparams.std_start, std_dev_end=self.hyperparams.std_end)
        # Construct the actor and critic networks and optimizers according to environment observation and action spaces:
        self.actor_critic_networks = ActorCriticNetworks(num_observations=self.num_observations, num_actions=self.num_actions, multivariate_gauss_dist=self.multivariate_gauss_dist,  lr=self.hyperparams.lr, learn_std=self.hyperparams.learn_std,  hidden_dim=64)
        # Create a rollout computer to abstract the computations needed.
        self.rollout_computer = RolloutComputer()
        # Create a simple logger:
        self.perf_logger = PerformanceLogger()

    def learn(self, total_timesteps, verbose=True):
        """ This is the method where learning magic happens. All the high level PPO logic happens here. I have abstracted detailed calculations inside the rollout_computer."""

        self.actor_critic_networks.total_timesteps = total_timesteps # this will be used to update learning rate by time.
        self.multivariate_gauss_dist.total_timesteps = total_timesteps # this will be used to update std by time.

        # Learn until total_timesteps is reached!
        timestep, rollout_no = 0, 0
        while timestep < total_timesteps:
            # Collect a rollout
            rollout = self.collect_a_rollout()

            # Convert required lists to tensors
            states_tensor, actions_tensor, initial_log_probs_tensor = self.rollout_computer.convert_list_to_tensor(rollout.states), self.rollout_computer.convert_list_to_tensor(rollout.actions), self.rollout_computer.convert_list_to_tensor(rollout.action_log_probs)
            #[len(rollout), num_states], [len(rollout), num_actions], [len(rollout)]

            # Calculate estimated Q(a,s) values for each state-action in the rollout
            monte_carlo_qas = self.rollout_computer.estimated_qas(next_states=rollout.next_states, rewards=rollout.rewards, dones=rollout.dones, truncateds=rollout.truncateds, critic=self.actor_critic_networks.critic, discount_factor=self.hyperparams.gamma) # grad_required False
            #[len(rollout)]

            # Calculate estimated values for each state in the rollout using critic
            V = self.rollout_computer.value_estimates(states=states_tensor, critic=self.actor_critic_networks.critic) # grad_required True but it will detached when calculating the Advantages
            #[len(rollout)]

            if self.hyperparams.advantage_calc == "classic":
                # Calculate the normalized advantage estimate for each state-action using the estimated_qas and V. Remember A(s,a) = Q(s,a) - V(s)
                A = self.rollout_computer.advantage_estimates(estimated_qas=monte_carlo_qas, estimated_values=V.detach(), normalized=True) # grad_require False
                #[len(rollout)]
            else:
                A = self.rollout_computer.gae(rewards=rollout.rewards, values=V.detach(), last_state_val=self.actor_critic_networks.critic(torch.from_numpy(rollout.next_states[-1]).float()).detach(),  dones=rollout.dones, gamma=self.hyperparams.gamma, gae_lambda=self.hyperparams.gae_lambda)

            timestep += len(rollout)  # update timestep
            rollout_no += 1  # update current rollout no

            # For each rollout, we'll be using the samples repeatedly to increase sample efficiency with the help of clipped ppo loss preventing destructive updates.
            for _ in range(self.hyperparams.n_updates_per_iteration):

                # Optionally obtain batches from the rollout
                batches = batchify(self.hyperparams.batch_size, states_tensor, actions_tensor, initial_log_probs_tensor, A, monte_carlo_qas) if self.hyperparams.batchify else [(states_tensor, actions_tensor, initial_log_probs_tensor, A, monte_carlo_qas)]

                # For each batch of transitions
                for states_tensor, actions_tensor, initial_log_probs_tensor, A, monte_carlo_qas in batches:
                    # At each update step I will recalculate the current action probs so that I can re calculate the importance sampling ratios.
                    curr_log_probs, entropies  = self.rollout_computer.curr_log_probs(states=states_tensor, actions=actions_tensor, actor=self.actor_critic_networks.actor, multivariate_gaussian_dist=self.multivariate_gauss_dist)
                    # [len(rollout)] grad_required True

                    # Find the ratio current s-a prob / initial s-a prob.
                    importance_sampling_ratios = torch.exp(curr_log_probs - initial_log_probs_tensor) # ln(curr_probs) - ln(initial_probs) is equal to ln(curr_probs/initial_probs) now if I apply exp() I will ge the ratios.
                    # Pay attention here Advantage estimates are fixed throughout the iterations. Those estimates help us reinforce required actions.
                    # Pay attention to the minus sign because we want to maximize the objective.
                    actor_loss = (-torch.min(importance_sampling_ratios*A, torch.clamp(importance_sampling_ratios, 1-self.hyperparams.clip, 1+self.hyperparams.clip)*A)).mean() - self.hyperparams.ent_coeff*(entropies.mean())

                    # Re-calculate the value estimates so that we can take another step re-utilizing the monte carlo return estimates.
                    V_curr = self.rollout_computer.value_estimates(states=states_tensor, critic=self.actor_critic_networks.critic) # grad_required True
                    critic_loss = nn.MSELoss()(V_curr, monte_carlo_qas) # here our target is the monte_carlo_qas so we won't update it!

                    # Update the actor and critic networks. Optionally use gradient clipping.
                    self.actor_critic_networks.take_a_gradient_step(actor_loss=actor_loss, critic_loss=critic_loss, clip_grad=self.hyperparams.clip_grad, max_grad=self.hyperparams.max_grad)
                # Check if current policy distribution is diverged too much from the initial one, if it is the case we'll stop using those samples to prevent destructive updates.
                if self.rollout_computer.new_policy_is_diverged(initial_log_probs_tensor, importance_sampling_ratios, curr_log_probs,self.hyperparams.max_kl_divergence):
                    print("kl diverged")
                    break

            # We can use learning rate annealing to stabilize learning by time.
            self.actor_critic_networks.reduce_learning_rate(curr_timestep=timestep, min_lr=self.hyperparams.min_lr)
            # Similarly we can update std deviation for the policy distribution to reduce exploration by time.
            self.multivariate_gauss_dist.update_cov_matrix(timestep=timestep)

            # Record rollout performance we'll be using them to obtain performance plots.
            self.perf_logger.add(avg_episodic_lengths=rollout.avg_episodic_lengths, avg_episodic_rewards=rollout.avg_episodic_rewards, timestep=timestep)
            if verbose:
                print(f"Rollout No: {rollout_no} - Timestep {timestep}/{total_timesteps}")
                print(f"Avg. Eps Len: {rollout.avg_episodic_lengths:.2f}, Avg. Eps Rew: {rollout.avg_episodic_rewards:.2f}")
                print(f"----------------------------------------------------")

        #self.perf_logger.plot()
        #self.actor_critic_networks.save()

    def collect_a_rollout(self):
        """ This method is used to collect a rollout (a set of trajectories) from the environment. After each rollout the algorithm
        will use the collected samples to update its actor and critic networks."""

        # Create a rollout buffer to store transitions and a rollout log to store episode lengths and rewards
        rollout_buffer = RolloutBuffer()

        # Play episodes until we collect enough samples for a rollout. A rollout can contain data from multiple episodes.
        t = 0
        eps_lens, eps_rews = [], []
        while t < self.hyperparams.rollout_len:
            s,_ = self.env.reset(seed=random.randint(0, 100000))
            episode_len, total_episode_reward = self.play_an_episode(s=s, rollout_buffer=rollout_buffer)
            eps_lens.append(episode_len)
            eps_rews.append(total_episode_reward)
            t += episode_len

        # Record rollout performance (avg episodic lengths and avg episodic rewards)
        rollout_buffer.avg_episodic_lengths = np.mean(eps_lens)
        rollout_buffer.avg_episodic_rewards = np.mean(eps_rews)

        return rollout_buffer

    def play_an_episode(self, s, rollout_buffer:RolloutBuffer) -> Tuple[int, float]:
        """ This method is used to collect transitions by playing a single episode starting from s. An episode can end when
        the environment returns Done, Truncated or the max_episode_len is reached."""

        episode_rewards = []

        for eps_t in range(self.hyperparams.max_episode_len):
            # Decide action (no grad calculated)
            action, log_prob_a = self.actor_critic_networks.sample_an_action(state=s)
            #action, log_prob_a = self.get_action(s)
            # Take a step in the environment
            s_next, r, done, truncated, _ = self.env.step(action)
            # Store the transitions later we'll be learning from those
            rollout_buffer.add_transition(state=s, action=action, action_log_prob=log_prob_a, next_state=s_next,reward=r, done=done, truncated=truncated)
            episode_rewards.append(r)
            # Update state:
            s = s_next
            # End episode when done is True
            if done or truncated: break

        # If episode ends due to max length but it is not done or truncated. Let's add truncated manually to make our value calculations more accurate.
        if eps_t + 1 >= self.hyperparams.max_episode_len and not rollout_buffer.dones[-1] and not rollout_buffer.truncateds[-1]:
            rollout_buffer.truncateds[-1] = True

        episode_length = eps_t +1
        episode_reward =sum(episode_rewards)

        return episode_length, episode_reward

    def seed_session(self, seed=None):
        """ This method is used to seed every random process to provide reproducibility."""
        if seed is None: seed = random.randint(0, 10000)
        self.seed = seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

