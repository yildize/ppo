import time
from injection.joystick import Joystick
from utils.render_wrapper import RenderWrapper

# THIS SCRIPT IS JUST TO TEST JOYSTICK ON MOUNTAINCAR ENV:

joystick = Joystick()
env = RenderWrapper('MountainCarContinuous-v0', normalize_obs=True)
env.activate_rendering()

n_episodes = 25  # Replace with the number of episodes you want to run

for episode in range(n_episodes):
    env.reset()
    done = False
    episode_reward = 0

    for _ in range(1000):
        time.sleep(0.02)  # Sleep to make the game run at a reasonable speed
        axis_0 = joystick.axis_0
        action = [axis_0]
        observation, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        if done:
            print(f"Episode {episode + 1}: Reward = {episode_reward}")
            break
env.close()
