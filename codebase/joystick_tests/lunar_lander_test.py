import gym
import time
import numpy as np
from injection.joystick import Joystick
from utils.render_wrapper import RenderWrapper

def stabilize_lander(observation, axis_steering, manual_control_intensity):
    """
    Stabilize the Lunar Lander based on its current angle and angular velocity.
    Less aggressive stabilization when there's manual control input.
    """
    angle, angular_velocity = observation[4], observation[5]

    # Adjust these thresholds and factors based on trial and error
    ANGLE_THRESHOLD = 0.05
    ANGULAR_VELOCITY_THRESHOLD = 0.1
    ANGLE_STABILIZATION_FACTOR = 1.0  # How aggressively to counteract the angle
    ANGULAR_VELOCITY_STABILIZATION_FACTOR = 1.0  # How aggressively to counteract the angular velocity

    stabilization = 0
    if abs(angle) > ANGLE_THRESHOLD:
        stabilization -= angle * ANGLE_STABILIZATION_FACTOR
    if abs(angular_velocity) > ANGULAR_VELOCITY_THRESHOLD:
        stabilization -= angular_velocity * ANGULAR_VELOCITY_STABILIZATION_FACTOR

    # Reduce stabilization intensity when there's manual control
    stabilization_intensity = 1 - abs(manual_control_intensity)
    stabilization *= stabilization_intensity

    # Combine user input with stabilization
    return np.clip(axis_steering + stabilization, -1, 1)

joystick = Joystick()
env = RenderWrapper('LunarLanderContinuous-v2', normalize_obs=True)
env.activate_rendering()

n_episodes = 25

for episode in range(n_episodes):
    observation,_ = env.reset()
    done = False
    episode_reward = 0

    for _ in range(1000):
        time.sleep(0.05)

        axis_thrust = -joystick.axis_1
        axis_steering = joystick.axis_0

        if axis_thrust < 0.03: axis_thrust = 0
        # Apply stabilization with consideration to manual control
        axis_steering = -stabilize_lander(observation, -axis_steering, axis_steering)

        action = [axis_thrust, axis_steering]

        observation, reward, done, truncated, info = env.step(action)
        episode_reward += reward

        if done or truncated:
            print(f"Episode {episode + 1}: Reward = {episode_reward}")
            break

env.close()
