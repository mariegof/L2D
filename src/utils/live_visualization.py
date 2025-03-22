import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import os

class LivePlotter:
    """Class for real-time visualization of training progress with best performance indicators"""
    
    def __init__(self, use_ipython=False, save_dir=None):
        self.use_ipython = use_ipython
        self.save_dir = save_dir
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Create figure and subplots
        self.fig, self.axs = plt.subplots(3, 1, figsize=(12, 15))
        self.fig.tight_layout(pad=3.0)
        
        # Initialize data storage
        self.episodes = []
        self.rewards = []
        self.weighted_sums = []
        self.losses = []
        self.validation_episodes = []
        self.validation_weighted_sums = []
        
        # Initialize best values trackers
        self.best_training_weighted_sum = float('inf')
        self.best_validation_weighted_sum = float('inf')
        
        # Initialize best value lines (initially invisible)
        self.best_training_line = self.axs[1].axhline(y=0, color='b', linestyle='--', alpha=0.5, visible=False)
        self.best_validation_line = self.axs[1].axhline(y=0, color='r', linestyle='--', alpha=0.5, visible=False)
        
        # Initialize main plots
        # Reward plot
        self.reward_line, = self.axs[0].plot([], [], 'b-', alpha=0.3, label='Training (per episode)')
        self.reward_avg_line, = self.axs[0].plot([], [], 'b-', linewidth=2, label='Training (smoothed)')
        self.axs[0].set_title('Training Reward')
        self.axs[0].set_xlabel('Episode')
        self.axs[0].set_ylabel('Reward')
        self.axs[0].legend()
        self.axs[0].grid(True, alpha=0.3)
        
        # Weighted Sum plot (main objective)
        self.weighted_sum_line, = self.axs[1].plot([], [], 'b-', alpha=0.3, label='Training (per episode)')
        self.weighted_sum_avg_line, = self.axs[1].plot([], [], 'b-', linewidth=2, label='Training (smoothed)')
        self.validation_line, = self.axs[1].plot([], [], 'r-o', linewidth=2, label='Validation')
        self.axs[1].set_title('Weighted Sum Objective: Training vs Validation')
        self.axs[1].set_xlabel('Episode')
        self.axs[1].set_ylabel('Weighted Sum')
        self.axs[1].legend()
        self.axs[1].grid(True, alpha=0.3)
        
        # Loss plot
        self.loss_line, = self.axs[2].plot([], [], 'g-', alpha=0.3, label='Training (per episode)')
        self.loss_avg_line, = self.axs[2].plot([], [], 'g-', linewidth=2, label='Training (smoothed)')
        self.axs[2].set_title('Loss During Training')
        self.axs[2].set_xlabel('Episode')
        self.axs[2].set_ylabel('Loss')
        self.axs[2].legend()
        self.axs[2].grid(True, alpha=0.3)
        
        plt.ion()  # Enable interactive mode
        self.fig.show()
    
    def update(self, episode, reward, weighted_sum, loss, 
               validation_episode=None, validation_weighted_sum=None):
        """Update plots with new data"""
        # Append data
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.weighted_sums.append(weighted_sum)
        self.losses.append(loss)
        
        # Calculate moving averages for smoothing
        window = min(100, len(self.rewards))
        if window > 1:
            reward_avg = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
            weighted_sum_avg = np.convolve(self.weighted_sums, np.ones(window)/window, mode='valid')
            loss_avg = np.convolve(self.losses, np.ones(window)/window, mode='valid')
            avg_episodes = self.episodes[window-1:]
        else:
            reward_avg = self.rewards
            weighted_sum_avg = self.weighted_sums
            loss_avg = self.losses
            avg_episodes = self.episodes
        
        # Update best training weighted sum
        if len(weighted_sum_avg) > 0:
            current_best_training = min(weighted_sum_avg)
            if current_best_training < self.best_training_weighted_sum:
                self.best_training_weighted_sum = current_best_training
                # Update best training line
                self.best_training_line.set_ydata([self.best_training_weighted_sum, self.best_training_weighted_sum])
                self.best_training_line.set_visible(True)
                self.best_training_line.set_label(f'Best training: {self.best_training_weighted_sum:.2f}')
        
        # Handle validation data if provided
        if validation_episode is not None and validation_weighted_sum is not None:
            self.validation_episodes.append(validation_episode)
            self.validation_weighted_sums.append(validation_weighted_sum)
            
            # Update best validation weighted sum
            if validation_weighted_sum < self.best_validation_weighted_sum:
                self.best_validation_weighted_sum = validation_weighted_sum
                # Update best validation line
                self.best_validation_line.set_ydata([self.best_validation_weighted_sum, self.best_validation_weighted_sum])
                self.best_validation_line.set_visible(True)
                self.best_validation_line.set_label(f'Best validation: {self.best_validation_weighted_sum:.2f}')
        
        # Update main plot lines
        self.reward_line.set_data(self.episodes, self.rewards)
        self.reward_avg_line.set_data(avg_episodes, reward_avg)
        
        self.weighted_sum_line.set_data(self.episodes, self.weighted_sums)
        self.weighted_sum_avg_line.set_data(avg_episodes, weighted_sum_avg)
        self.validation_line.set_data(self.validation_episodes, self.validation_weighted_sums)
        
        self.loss_line.set_data(self.episodes, self.losses)
        self.loss_avg_line.set_data(avg_episodes, loss_avg)
        
        # Refresh legends to show updated best values
        self.axs[1].legend()
        
        # Adjust axes limits
        for i, ax in enumerate(self.axs):
            ax.relim()
            ax.autoscale_view()
        
        # Redraw
        if self.use_ipython:
            clear_output(wait=True)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def save_final_plots(self):
        """Save final plots"""
        if not self.save_dir:
            return
        
        plt.ioff()  # Disable interactive mode for saving
        
        # Save combined plot
        self.fig.savefig(os.path.join(self.save_dir, 'training_curves.png'))
        
        # Save individual plots
        for i, (ax, name) in enumerate(zip(self.axs, ['reward', 'weighted_sum', 'loss'])):
            fig, ax_new = plt.subplots(figsize=(10, 6))
            lines = ax.get_lines()
            for line in lines:
                if line.get_visible():
                    ax_new.plot(line.get_xdata(), line.get_ydata(), 
                               label=line.get_label(), color=line.get_color(),
                               marker=line.get_marker(), linestyle=line.get_linestyle())
            
            ax_new.set_title(ax.get_title())
            ax_new.set_xlabel(ax.get_xlabel())
            ax_new.set_ylabel(ax.get_ylabel())
            ax_new.legend()
            ax_new.grid(True, alpha=0.3)
            
            fig.savefig(os.path.join(self.save_dir, f'{name}_curve.png'))
            plt.close(fig)
        
        plt.ion()  # Re-enable interactive mode