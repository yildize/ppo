import time
from injection.joystick import Joystick
from utils.render_wrapper import RenderWrapper

# THIS SCRIPT IS JUST TO TEST JOYSTICK ON CAR RACING ENVIRONMENT

joystick = Joystick()
env = RenderWrapper('CarRacing-v2', normalize_obs=True)
env.activate_rendering()

n_episodes = 25  # Replace with the number of episodes you want to run

for episode in range(n_episodes):
    env.reset()
    done = False
    episode_reward = 0

    for _ in range(1000):
        time.sleep(0.02)  # Sleep to make the game run at a reasonable speed
        # Assuming the joystick provides axes for steering, gas, and brake
        axis_steering = joystick.axis_0  # Replace with the actual axis index for steering
        axis_gas = -joystick.axis_1       # Replace with the actual axis index for gas
        axis_brake = abs(axis_gas) if axis_gas < 0 else 0 #joystick.axis_2     # Replace with the actual axis index for brake
        # Action vector for CarRacing-v2 [steering, gas, brake]
        action = [axis_steering ,axis_gas , axis_brake]#[axis_steering, axis_gas, axis_brake]
        # Take the action and observe the new state and reward
        observation, reward, done, truncated, info = env.step(action)
        # Accumulate the reward
        episode_reward += reward
        if done or truncated:
            # Print the reward for the episode when it's done
            print(f"Episode {episode + 1}: Reward = {episode_reward}")
            break

env.close()  # Close the environment
