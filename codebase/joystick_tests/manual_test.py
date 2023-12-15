import gym
import pygame
import time
import numpy as np
from injection.joystick import Joystick
from utils.render_wrapper import RenderWrapper

joystick = Joystick()
# Initialize the environment
env = RenderWrapper('MountainCarContinuous-v0', normalize_obs=True)
env.activate_rendering()

n_episodes = 25  # Replace with the number of episodes you want to run

for episode in range(n_episodes):
    observation,_ = env.reset()
    done = False
    episode_reward = 0

    for _ in range(1000):
        time.sleep(0.02)  # Sleep to make the game run at a reasonable speed
        # Assuming your joystick has at least one axis for horizontal movement
        # Axis 0 for left/right movement
        axis_0 = joystick.axis_0

        # The action is a single number, the force to apply left or right
        # We map the left/right movement to force
        # Depending on the physical setup, you might need to invert the axis with `-axis_0`
        action = [axis_0]

        # Take the action and observe the new state and reward
        observation, reward, done, truncated, info = env.step(action)

        # Accumulate the reward
        episode_reward += reward

        if done:
            # Print the reward for the episode when it's done
            print(f"Episode {episode + 1}: Reward = {episode_reward}")
            break

env.close()  # Close the environment
