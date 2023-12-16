import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Create and Wrap the Environment
env_name = "MountainCarContinuous-v0"
env = make_vec_env(env_name, n_envs=1)
env = VecNormalize(env, norm_obs=True)

# Initialize the Agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the Agent
total_timesteps = 1_000_000  # Adjust this as needed
model.learn(total_timesteps=total_timesteps)

# Save the Model and Environment Statistics
#model.save("./injection/assistive_actors/assistant_trainings/ppo_mountaincar")
#env.save("/injection/assistive_actors/assistant_trainings/ppo_mountaincar_env_stats.pkl")

# To load the model and environment statistics later
#loaded_model = PPO.load("/injection/assistive_actors/assistant_trainings/ppo_mountaincar")
#loaded_env = VecNormalize.load("/injection/assistive_actors/assistant_trainings/ppo_mountaincar_env_stats.pkl", env)