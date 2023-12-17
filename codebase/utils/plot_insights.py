import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class InsightPlots:
    """ This class help printing insightful plots helping debug the implementation"""
    def __init__(self, A, states_tensor, unnormalized_states_tensor, actions_tensor, initial_log_probs_tensor, actor):
        # Separate the data based on negative and positive velocities
        self.A = A
        self.states_tensor = states_tensor
        self.actions_tensor = actions_tensor
        self.initial_log_probs = initial_log_probs_tensor
        self.actor = actor

        # Unnormalized states tensor is just will be used to obtain indexes
        if unnormalized_states_tensor is None: unnormalized_states_tensor = states_tensor

        self.neg_vels = unnormalized_states_tensor[:, 1] < 0
        self.pos_vels = unnormalized_states_tensor[:, 1] >= 0

        self.neg_vel_states = states_tensor[self.neg_vels]
        self.pos_vel_states = states_tensor[self.pos_vels]

        self.neg_vel_advantages = A[self.neg_vels]
        self.pos_vel_advantages = A[self.pos_vels]

        self.neg_vel_actions = actions_tensor[self.neg_vels]
        self.pos_vel_actions = actions_tensor[self.pos_vels]

        self.neg_vel_log_probs = initial_log_probs_tensor[self.neg_vels]
        self.pos_vel_log_probs = initial_log_probs_tensor[self.pos_vels]

        # Convert log probabilities to probabilities
        self.neg_vel_probs = torch.exp(self.neg_vel_log_probs)
        self.pos_vel_probs = torch.exp(self.pos_vel_log_probs)

    def plot_all(self):
        self.plot_advantage_comparison_box()
        #self.plot_predicted_mean_actions()
        #self.plot_states_by_advantage()
        self.plot_mean_actions_and_advantages()
        self.plot_state_distribution_colored()



    def plot_advantage_comparison_box(self):
        # Data preparation
        neg_advantages = self.neg_vel_advantages.numpy()
        pos_advantages = self.pos_vel_advantages.numpy()

        # Calculating IQR for negative and positive velocity advantages
        neg_Q1, neg_Q3 = np.percentile(neg_advantages, [25, 75])
        pos_Q1, pos_Q3 = np.percentile(pos_advantages, [25, 75])
        neg_IQR = neg_Q3 - neg_Q1
        pos_IQR = pos_Q3 - pos_Q1

        # Calculating the number of outliers and total samples
        neg_outliers = np.sum((neg_advantages < neg_Q1 - 1.5 * neg_IQR) | (neg_advantages > neg_Q3 + 1.5 * neg_IQR))
        pos_outliers = np.sum((pos_advantages < pos_Q1 - 1.5 * pos_IQR) | (pos_advantages > pos_Q3 + 1.5 * pos_IQR))
        total_neg = len(neg_advantages)
        total_pos = len(pos_advantages)

        # Plotting the boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[neg_advantages, pos_advantages], notch=True, palette="Set2")
        plt.xticks([0, 1], ['Negative Velocity', 'Positive Velocity'])

        # Annotating the plot with IQR, outlier information, and total number of samples
        plt.text(-0.35, neg_Q3 + 1.5 * neg_IQR, f'#Outliers: {neg_outliers}\n#Total: {total_neg}', verticalalignment='bottom') #IQR: {neg_IQR:.2f}\n
        plt.text(0.65, pos_Q3 + 1.5 * pos_IQR, f'#Outliers: {pos_outliers}\n#Total: {total_pos}', verticalalignment='bottom') #IQR: {pos_IQR:.2f}\n

        plt.title('Comparison of Advantage Values')
        plt.ylabel('Advantage')
        plt.show()


    def plot_mean_actions_and_advantages(self):
        # Ensure no gradients are computed
        with torch.no_grad():
            # Get predicted mean actions and Advantage values for all states
            predicted_means, _ = self.actor(self.states_tensor)
            advantages = self.A.detach()

        # Convert to numpy for plotting
        predicted_means = predicted_means.numpy()
        advantages = advantages.numpy()

        # Calculate the percentages of negative and positive velocity states
        total_states = len(self.states_tensor)
        neg_percentage = (self.neg_vels.sum().item() / total_states) * 100
        pos_percentage = (self.pos_vels.sum().item() / total_states) * 100

        # Set up the figure and axes for the subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

        # Plot 1: Predicted Mean Actions
        sc1 = axes[0].scatter(self.states_tensor[:, 0].numpy(), self.states_tensor[:, 1].numpy(), c=predicted_means, cmap='viridis', alpha=0.7)
        fig.colorbar(sc1, ax=axes[0], label='Predicted Mean Action')
        axes[0].set_title('Predicted Mean Actions at Different States')
        axes[0].set_xlabel('Position')
        axes[0].set_ylabel('Velocity')
        axes[0].annotate(f'Negative Velocity States: {neg_percentage:.2f}%', xy=(0.01, 0.99), xycoords='axes fraction', fontsize=10, ha='left', va='top')
        axes[0].annotate(f'Positive Velocity States: {pos_percentage:.2f}%', xy=(0.01, 0.94), xycoords='axes fraction', fontsize=10, ha='left', va='top')

        # Plot 2: States Colored by Advantage Values
        sc2 = axes[1].scatter(self.states_tensor[:, 0].numpy(), self.states_tensor[:, 1].numpy(), c=advantages, cmap='plasma', alpha=0.7)
        fig.colorbar(sc2, ax=axes[1], label='Advantage Value')
        axes[1].set_title('States Colored by Advantage Values')
        axes[1].set_xlabel('Normalized Position')
        axes[1].set_ylabel('Normalized Velocity')

        # Adjust the layout and display the plots
        plt.tight_layout()
        plt.show()


    def plot_predicted_mean_actions(self):
        """Plot the predicted mean actions from the actor network at different states,
           and annotate with the percentage of positive and negative velocity states."""
        with torch.no_grad():  # Ensure no gradients are computed during prediction
            # Get predicted mean actions for all states
            predicted_means, _ = self.actor(self.states_tensor)

        # Detach predictions and convert to numpy for plotting
        predicted_means = predicted_means.detach().numpy()

        # Calculate the percentages of negative and positive velocity states
        total_states = len(self.states_tensor)
        neg_percentage = (self.neg_vels.sum().item() / total_states) * 100
        pos_percentage = (self.pos_vels.sum().item() / total_states) * 100

        plt.figure(figsize=(10, 6))
        # Scatter plot where color intensity represents the magnitude of the predicted mean action
        plt.scatter(self.states_tensor[:, 0].numpy(), self.states_tensor[:, 1].numpy(), c=predicted_means, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Predicted Mean Action')
        plt.title('Predicted Mean Actions at Different States')
        plt.xlabel('Position')
        plt.ylabel('Velocity')

        # Annotate with the percentage of negative and positive velocity states
        plt.annotate(f'Negative Velocity States: {neg_percentage:.2f}%', xy=(0.01, 0.99), xycoords='axes fraction', fontsize=10, ha='left', va='top', backgroundcolor='white')
        plt.annotate(f'Positive Velocity States: {pos_percentage:.2f}%', xy=(0.01, 0.94), xycoords='axes fraction', fontsize=10, ha='left', va='top', backgroundcolor='white')

        plt.show()

    def plot_states_by_advantage(self):
        """Plot the states colored by their corresponding Advantage values."""
        with torch.no_grad():  # We do not need gradients for this operation
            # Detach the Advantage values and convert to numpy for plotting
            advantages = self.A.detach().numpy()

        plt.figure(figsize=(10, 6))
        # Scatter plot where color intensity represents the magnitude of Advantage values
        scatter = plt.scatter(self.states_tensor[:, 0].numpy(), self.states_tensor[:, 1].numpy(), c=advantages, cmap='plasma', alpha=0.7)
        plt.colorbar(scatter, label='Advantage Value')
        plt.title('States Colored by Advantage Values')
        plt.xlabel('Normalized Position')
        plt.ylabel('Normalized Velocity')
        plt.show()


    def plot_state_distribution(self):
        """Plot the distribution of all states during the rollout."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.states_tensor[:, 0].numpy(), y=self.states_tensor[:, 1].numpy(), alpha=0.5)
        plt.title('Rollout State Distribution')
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.show()

    def plot_state_distribution_with_counts(self):
        """Plot the state distribution and annotate with the number of positive and negative velocity states."""
        plt.figure(figsize=(12, 8))

        # Scatter plot of all states
        sns.scatterplot(x=self.states_tensor[:, 0].numpy(), y=self.states_tensor[:, 1].numpy(), alpha=0.5)

        # Annotating the plot with the number of negative and positive velocity states
        neg_count = self.neg_vels.sum().item()  # Summing the True values (which are treated as 1)
        pos_count = self.pos_vels.sum().item()  # Summing the True values (which are treated as 1)
        plt.annotate(f'Negative Velocity States: {neg_count}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                     backgroundcolor='white')
        plt.annotate(f'Positive Velocity States: {pos_count}', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12,
                     backgroundcolor='white')

        plt.title('State Distribution with Velocity Counts')
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.show()

    def plot_state_distribution_colored(self):
        """Plot the distribution of all states during the rollout."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.neg_vel_states[:, 0].numpy(), y=self.neg_vel_states[:, 1].numpy(), alpha=0.5)
        sns.scatterplot(x=self.pos_vel_states[:, 0].numpy(), y=self.pos_vel_states[:, 1].numpy(), alpha=0.5)
        plt.title('Rollout State Distribution')
        plt.xlabel('Normalized Position')
        plt.ylabel('Normalized Velocity')
        plt.show()



    def plot_advantage_distribution(self):
        """Plot the distribution of Advantage values for negative and positive velocity states."""
        plt.figure(figsize=(10, 6))

        # Prepare data for plotting
        data = [self.neg_vel_advantages.numpy(), self.pos_vel_advantages.numpy()]
        labels = ['Negative Velocity', 'Positive Velocity']

        # Using a boxplot to compare distributions
        sns.boxplot(data=data)
        plt.xticks(range(len(labels)), labels)
        plt.title('Advantage Values Distribution')
        plt.ylabel('Advantage Value')
        plt.show()

        # Alternatively, using histograms
        plt.figure(figsize=(10, 6))
        plt.hist(self.neg_vel_advantages.numpy(), bins=30, alpha=0.7, label='Negative Velocity')
        plt.hist(self.pos_vel_advantages.numpy(), bins=30, alpha=0.7, label='Positive Velocity')
        plt.title('Advantage Values Distribution')
        plt.xlabel('Advantage Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


    def actions_taken(self):
        #Comparing Advantage Values for Negative and Positive Velocity States
        plt.figure(figsize=(10, 6))
        plt.hist([self.neg_vel_actions.numpy().flatten(), self.pos_vel_actions.numpy().flatten()],
                 bins=30, alpha=0.7, label=['Negative Velocity', 'Positive Velocity'])
        plt.title('Actions Distribution for Negative and Positive Velocity States')
        plt.xlabel('Action Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    def action_probs(self):
        # Probabilities of Actions in Negative and Positive Velocity States
        plt.figure(figsize=(10, 6))
        plt.hist([self.neg_vel_probs.numpy(), self.pos_vel_probs.numpy()], bins=30, alpha=0.7,
                 label=['Negative Velocity', 'Positive Velocity'])
        plt.title('Action Probabilities for Negative and Positive Velocity States')
        plt.xlabel('Probability of Action')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    def action_probs_actor(self):
        neg_vel_actions_pred, _ = self.actor(self.neg_vel_states)
        pos_vel_actions_pred, _ = self.actor(self.pos_vel_states)

        # Detach the predictions from the computation graph and convert to numpy for plotting
        neg_vel_actions_pred = neg_vel_actions_pred.detach().numpy()
        pos_vel_actions_pred = pos_vel_actions_pred.detach().numpy()

        # Plotting

        # Scatter plot of actor predictions for negative and positive velocities
        plt.figure(figsize=(10, 6))
        plt.scatter(self.neg_vel_states[:, 0].numpy(), neg_vel_actions_pred, label='Negative Velocity', alpha=0.1)
        plt.scatter(self.pos_vel_states[:, 0].numpy(), pos_vel_actions_pred, label='Positive Velocity', alpha=0.1)
        plt.title('Actor Predictions for Negative and Positive Velocity States')
        plt.xlabel('Position')
        plt.ylabel('Predicted Mean Action')
        plt.legend()
        plt.show()



"""
# Assuming these tensors are already defined:
# states_tensor, actions_tensor, initial_log_probs_tensor, A

# Separate the data based on negative and positive velocities
neg_vels = states_tensor[:, 1] < 0
pos_vels = states_tensor[:, 1] >= 0

neg_vel_states = states_tensor[neg_vels]
pos_vel_states = states_tensor[pos_vels]

neg_vel_advantages = A[neg_vels]
pos_vel_advantages = A[pos_vels]

neg_vel_actions = actions_tensor[neg_vels]
pos_vel_actions = actions_tensor[pos_vels]

neg_vel_log_probs = initial_log_probs_tensor[neg_vels]
pos_vel_log_probs = initial_log_probs_tensor[pos_vels]

# Convert log probabilities to probabilities
neg_vel_probs = torch.exp(neg_vel_log_probs)
pos_vel_probs = torch.exp(pos_vel_log_probs)

# Plotting

# 1. Comparing Advantage Values for Negative and Positive Velocity States
plt.figure(figsize=(10, 6))
sns.boxplot(data=[neg_vel_advantages.numpy(), pos_vel_advantages.numpy()], 
            notch=True, 
            palette="Set2")
plt.xticks([0, 1], ['Negative Velocity', 'Positive Velocity'])
plt.title('Comparison of Advantage Values')
plt.ylabel('Advantage')
plt.show()
"""



"""
# 2. Actions Taken in Negative and Positive Velocity States
plt.figure(figsize=(10, 6))
plt.hist([neg_vel_actions.numpy().flatten(), pos_vel_actions.numpy().flatten()], 
         bins=30, alpha=0.7, label=['Negative Velocity', 'Positive Velocity'])
plt.title('Actions Distribution for Negative and Positive Velocity States')
plt.xlabel('Action Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
"""


"""
# 3. Probabilities of Actions in Negative and Positive Velocity States
plt.figure(figsize=(10, 6))
plt.hist([neg_vel_probs.numpy(), pos_vel_probs.numpy()], bins=30, alpha=0.7, label=['Negative Velocity', 'Positive Velocity'])
plt.title('Action Probabilities for Negative and Positive Velocity States')
plt.xlabel('Probability of Action')
plt.ylabel('Frequency')
plt.legend()
plt.show()
"""


"""
# Assuming the actor network is accessed as self.actor_critic_networks.actor
# and it returns a tuple (mean action, standard deviation) for given states

# Get actor predictions for negative and positive velocity states
neg_vel_actions_pred, _ = self.actor_critic_networks.actor(neg_vel_states)
pos_vel_actions_pred, _ = self.actor_critic_networks.actor(pos_vel_states)

# Detach the predictions from the computation graph and convert to numpy for plotting
neg_vel_actions_pred = neg_vel_actions_pred.detach().numpy()
pos_vel_actions_pred = pos_vel_actions_pred.detach().numpy()

# Plotting

# Scatter plot of actor predictions for negative and positive velocities
plt.figure(figsize=(10, 6))
plt.scatter(neg_vel_states[:, 0].numpy(), neg_vel_actions_pred, label='Negative Velocity', alpha=0.7)
plt.scatter(pos_vel_states[:, 0].numpy(), pos_vel_actions_pred, label='Positive Velocity', alpha=0.7)
plt.title('Actor Predictions for Negative and Positive Velocity States')
plt.xlabel('Position')
plt.ylabel('Predicted Mean Action')
plt.legend()
plt.show()
"""


"""
import seaborn as sns
neg_vels = states_tensor[:, 1] < 0
pos_vels = states_tensor[:, 1] >= 0

neg_vel_states = states_tensor[neg_vels]
pos_vel_states = states_tensor[pos_vels]

neg_vel_advantages = A[neg_vels]
pos_vel_advantages = A[pos_vels]

neg_vel_actions = actions_tensor[neg_vels]
pos_vel_actions = actions_tensor[pos_vels]

neg_vel_log_probs = initial_log_probs_tensor[neg_vels]
pos_vel_log_probs = initial_log_probs_tensor[pos_vels]

# Convert log probabilities to probabilities
neg_vel_probs = torch.exp(neg_vel_log_probs)
pos_vel_probs = torch.exp(pos_vel_log_probs)

neg_vel_actions_pred, _ = self.actor_critic_networks.actor(neg_vel_states)
pos_vel_actions_pred, _ = self.actor_critic_networks.actor(pos_vel_states)

# Detach the predictions from the computation graph and convert to numpy for plotting
neg_vel_actions_pred = neg_vel_actions_pred.detach().numpy()
pos_vel_actions_pred = pos_vel_actions_pred.detach().numpy()

# Plotting

# Scatter plot of actor predictions for negative and positive velocities
plt.figure(figsize=(10, 6))
plt.scatter(neg_vel_states[:, 0].numpy(), neg_vel_actions_pred, label='Negative Velocity', alpha=0.1)
plt.scatter(pos_vel_states[:, 0].numpy(), pos_vel_actions_pred, label='Positive Velocity', alpha=0.1)
plt.title('Actor Predictions for Negative and Positive Velocity States')
plt.xlabel('Position')
plt.ylabel('Predicted Mean Action')
plt.legend()
plt.show()

"""