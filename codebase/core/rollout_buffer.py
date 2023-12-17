import numpy as np
from typing import List, Tuple


class RolloutBuffer:
    """ This is a data container utility class. I will use it to store rollout data. This data will be used to
    update the networks."""

    def __init__(self):
        self.states:List[np.ndarray] = [] # (rollout_len, dimension of observation)
        self.unnormalized_states: List[np.ndarray] = []  # (rollout_len, dimension of observation)
        self.actions:List[np.ndarray] = [] # (rollout_len, dimension of action)
        self.action_log_probs:List[float] = [] # (rollout_len)
        self.next_states:List[np.ndarray] = [] # (rollout_len)
        self.rewards:List[float] = [] # (number of episodes, number of timesteps per episode)
        self.dones:List[bool] = [] # (rollout_len)
        self.truncateds:List[bool] = [] #(rollout_len)

        self.avg_episodic_rewards = None
        self.avg_episodic_lengths = None

    def add_transition(self, state:np.ndarray, action:np.ndarray, action_log_prob:float, next_state:np.ndarray,
                       reward:float, done:bool, truncated:bool):
        """ This method adds a transition to the buffer. This will be called after each step of interaction with the
        environment during the rollout."""
        self.states.append(state)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.truncateds.append(truncated)

    def __len__(self):
        return len(self.states)