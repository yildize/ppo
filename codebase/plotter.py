import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import List, Tuple
from utils import PerformanceLogger


class Plotter:
    """ This class simply plots the provided PPO training logs. Since I will be providing logs consisting of different
    training sessions with different seeds. Each of them can be in different shape, therefore I apply a interpolation
    operation before plotting. Also, I will be calculating the mean and stds to provide a more statistically
    meaningful plot."""

    @staticmethod
    def calc_interpolated_rewards(logs, common_timesteps)->List[np.ndarray]:
        """ This method obtains interpolated rewards using the raw log objects. In that operation to provide
        extrapolation errors I only consider a common index range for all logs."""
        interpolated_rewards = []
        for log in logs:
            # Only consider the part of the log that is within the common range
            valid_indices = (log.timesteps >= common_timesteps[0]) & (log.timesteps <= common_timesteps[-1])
            valid_timesteps = np.array(log.timesteps)[valid_indices]
            valid_rewards = np.array(log.avg_eps_rews)[valid_indices]
            # Create an interpolation function for the current log
            interp_func = interp1d(valid_timesteps, valid_rewards, kind='linear', bounds_error=False, fill_value='extrapolate')
            # Use the function to interpolate the rewards
            interpolated_reward = interp_func(common_timesteps)
            interpolated_rewards.append(interpolated_reward)

        return interpolated_rewards

    @staticmethod
    def calc_mean_and_stds(interpolated_rewards:List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """ It calculates mean and std from the interpolated rewards."""
        # Now calculate the mean and standard deviation at each common timestep
        mean_rewards = np.mean(interpolated_rewards, axis=0)
        std_rewards = np.std(interpolated_rewards, axis=0)
        return mean_rewards, std_rewards

    @staticmethod
    def plot(logs:List[PerformanceLogger], number_of_points:int, title:str, save_path:str):
        """ This method will be the plotting the given logs with the provided resolution (number of points). It will first
        obtain a same length rewards using interpolation, calculate mean and std using those and finally plotting them."""
        # Find the common range of timesteps
        min_timestep = max(log.timesteps[0] for log in logs)  # the first timestep after the initial rollout for the run with the latest start
        max_timestep = min(log.timesteps[-1] for log in logs)  # the last timestep for the run with the earliest finish

        # Generate common timesteps within this range
        common_timesteps = np.linspace(min_timestep, max_timestep, num=number_of_points) # number_of_points evenly spaced points

        interpolated_rewards = Plotter.calc_interpolated_rewards(logs=logs, common_timesteps=common_timesteps)
        mean_rewards, std_rewards = Plotter.calc_mean_and_stds(interpolated_rewards)
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(common_timesteps, mean_rewards, label='Mean Episodic Reward')
        plt.fill_between(common_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label='Std Dev')
        # Set larger font size for y-ticks
        plt.yticks(fontsize=15)  # You can adjust the size as needed
        # Annotate the final y-value
        final_y_value = mean_rewards[-1]
        plt.annotate(f'{final_y_value:.2f}',  # Formatting to 2 decimal places
                     xy=(common_timesteps[-1], final_y_value),
                     xytext=(10, 0),  # This offsets the text slightly to the right
                     textcoords='offset points',
                     ha='center',
                     va='bottom',
                     fontsize=12)  # Adjust font size as needed
        plt.title(title)
        plt.xlabel('Timesteps')
        plt.ylabel('Average Rollout Reward')
        plt.legend()
        plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.savefig(save_path)



