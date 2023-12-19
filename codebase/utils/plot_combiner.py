import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List


# A function to load the data from a pickle file
def load_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data



# Define a function to plot the comparison
def plot_multiple_comparisons_with_injections(data_list, names, colors, injections, title, save_path, zoom_start=0):
    """
    Plot multiple sets of data on the same graph, including mean, standard deviation, and injection periods.

    :param data_list: List of data dictionaries, each containing 'common_timesteps', 'mean_rewards', 'std_rewards'
    :param names: List of names for the legend labels corresponding to each data set
    :param colors: List of colors for each data set
    :param injections: List of tuples, each containing (injection_start_timestep, injection_end_timestep, label_str)
    :param title: The title of the plot
    :param save_path: Path where to save the plot image
    """
    if len(data_list) != len(names) or len(names) != len(colors):
        raise ValueError("Length of data_list, names, and colors must be the same.")

    plt.figure(figsize=(12, 6))

    for data, name, color in zip(data_list, names, colors):
        # Filter data for timesteps greater than zoom_start
        filtered_indices = [i for i, x in enumerate(data['common_timesteps']) if x >= zoom_start]
        filtered_timesteps = [data['common_timesteps'][i] for i in filtered_indices]
        filtered_mean_rewards = [data['mean_rewards'][i] for i in filtered_indices]
        filtered_std_rewards = [data['std_rewards'][i] for i in filtered_indices]

        # Plot mean rewards and standard deviation area for the zoomed-in data
        plt.plot(filtered_timesteps, filtered_mean_rewards, label=f'{name}', color=color)
        plt.fill_between(filtered_timesteps,
                         [m - s for m, s in zip(filtered_mean_rewards, filtered_std_rewards)],
                         [m + s for m, s in zip(filtered_mean_rewards, filtered_std_rewards)], alpha=0.2, color=color)

        # Annotate the final y-value for the zoomed-in data
        if filtered_indices:
            final_y_value = filtered_mean_rewards[-1]
            plt.annotate(f'{final_y_value:.2f}',
                         xy=(filtered_timesteps[-1], final_y_value),
                         xytext=(10, 0),
                         textcoords='offset points',
                         ha='center',
                         va='bottom', #if color=="GREEN" else "top",
                         fontsize=12,
                         color=color)

    # Add vertical lines for injections within the zoomed-in range
    for start, end, label in injections:
        if start >= zoom_start:
            plt.axvline(x=start, color='k', linestyle=':', alpha=0.7)
            plt.axvline(x=end, color='k', linestyle=':', alpha=0.7)
            plt.text((start + end) / 2, plt.ylim()[1] * 0.95, label, horizontalalignment='center', verticalalignment='top', color='k', alpha=0.7, rotation=90)

    plt.title(title)
    plt.xlabel('Timesteps')
    plt.ylabel('Average Rollout Reward')
    plt.legend()
    plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.savefig(save_path)
    plt.show()

    return save_path


# Now that we have the correct files, let's load the data from these new files
injective_ppo_file_path = 'D:\EBackup\ppo-injection\ppo\codebase\logs\LUNARLANDER300K\perf_injective_ppo.pkl'
ppo_file_path = 'D:\EBackup\ppo-injection\ppo\codebase\logs\LUNARLANDER300K\perf_ppo.pkl'
#assistive_file_path = 'D:\EBackup\ppo-injection\ppo\codebase\logs\INJECTION-EFFECT-300K\\assistive-data.pkl'

#injective_ppo_file_path

# Generate the plot with refactored function including std devs
plot_multiple_comparisons_with_std_path = plot_multiple_comparisons_with_injections(
    data_list=[load_data(injective_ppo_file_path), load_data(ppo_file_path), ],#load_data(assistive_file_path)],
    names=["Injective-PPO", "PPO"],
    colors=["GREEN", "RED"],
    injections=[
        # (5000, 7000, ''),
        # (10000, 12000, ''),
        # (20000, 22000, ''),
        # (36000, 38000, ''),
        # (55000, 57000, ''),
        # (75000, 77000, ''),
        # (101000, 103000, ''),
        # (150000, 15200, ''),
        # (155000, 157000, ''),
        # (190000, 192000, ''),
        # (197000, 199000, ''),
        # (197000, 199000, ''),
        # (220000, 221000, ''),
        # (223000, 224000, ''),
        # (245000, 247000, ''),
        # (248000, 249000, ''),

    ],
    title="Peformance Comparison",
    save_path='D:\EBackup\ppo-injection\ppo\codebase\logs\INJECTION-EFFECT-300K\compare.png'
)
#
