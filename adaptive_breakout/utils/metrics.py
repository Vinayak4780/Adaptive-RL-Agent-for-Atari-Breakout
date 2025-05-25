import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2
import os
from datetime import datetime
import time
import gym

class MetricsTracker:
    """
    Class for tracking and visualizing training metrics
    """
    def __init__(self, log_dir=None):
        """
        Initialize the metrics tracker.
        
        Args:
            log_dir: Directory to save logs and visualizations
        """
        if log_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = f"logs_{timestamp}"
            
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics dictionaries
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'mean_q_values': [],
            'losses': [],
            'dynamics_losses': [],
            'epsilon_values': [],
            'curriculum_levels': [],
            'env_changes': [],
            'recovery_times': []
        }
        
        # For tracking rewards across difficulty changes
        self.rewards_by_difficulty = {}
        self.start_time = time.time()
    
    def add_metric(self, metric_name, value):
        """Add a value to a specific metric"""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
        else:
            self.metrics[metric_name] = [value]
    
    def log_episode(self, episode_idx, episode_reward, episode_length, mean_q_value, 
                   loss, dynamics_loss, epsilon, curriculum_level, env_changes=None):
        """Log metrics for a completed episode"""
        self.add_metric('episode_rewards', episode_reward)
        self.add_metric('episode_lengths', episode_length)
        self.add_metric('mean_q_values', mean_q_value)
        self.add_metric('losses', loss)
        self.add_metric('dynamics_losses', dynamics_loss)
        self.add_metric('epsilon_values', epsilon)
        self.add_metric('curriculum_levels', curriculum_level)
        
        if env_changes:
            self.add_metric('env_changes', env_changes)
        
        # Log to console
        elapsed_time = time.time() - self.start_time
        print(f"Episode {episode_idx} | " 
              f"Reward: {episode_reward:.1f} | "
              f"Length: {episode_length} | "
              f"Mean Q: {mean_q_value:.3f} | "
              f"Loss: {loss:.5f} | "
              f"Dynamics Loss: {dynamics_loss:.5f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Curriculum: {curriculum_level} | "
              f"Time: {elapsed_time:.1f}s")
        
        # Save metrics periodically
        if episode_idx % 10 == 0:
            self.save_metrics()
    
    def log_difficulty_performance(self, difficulty_params, reward):
        """Log performance metrics for specific difficulty settings"""
        # Create a key based on difficulty parameters
        key = (
            f"paddle_{difficulty_params['paddle_speed']:.1f}_"
            f"ball_{difficulty_params['ball_speed']:.1f}_"
            f"size_{difficulty_params['paddle_size']:.1f}"
        )
        
        if key not in self.rewards_by_difficulty:
            self.rewards_by_difficulty[key] = []
        
        self.rewards_by_difficulty[key].append(reward)
    
    def log_recovery_time(self, frames_to_recover):
        """Log time (in frames) to recover after a difficulty change"""
        self.add_metric('recovery_times', frames_to_recover)
    
    def save_metrics(self):
        """Save all metrics to disk"""
        np.save(f"{self.log_dir}/metrics.npy", self.metrics)
        
        # Also save rewards by difficulty
        np.save(f"{self.log_dir}/rewards_by_difficulty.npy", self.rewards_by_difficulty)
    
    def plot_metrics(self, save=True, window_size=10):
        """Plot training metrics with a rolling window for smoothing"""
        plt.figure(figsize=(20, 12))
        
        # Plot episode rewards
        plt.subplot(2, 3, 1)
        rewards = self.metrics['episode_rewards']
        plt.plot(rewards, alpha=0.3, color='blue')
        # Add smoothed line
        if len(rewards) > window_size:
            smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(rewards)), smoothed, color='blue')
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot mean Q values
        plt.subplot(2, 3, 2)
        q_values = self.metrics['mean_q_values']
        plt.plot(q_values, alpha=0.3, color='green')
        if len(q_values) > window_size:
            smoothed = np.convolve(q_values, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(q_values)), smoothed, color='green')
        plt.title('Mean Q Values')
        plt.xlabel('Episode')
        plt.ylabel('Q Value')
        
        # Plot losses
        plt.subplot(2, 3, 3)
        losses = self.metrics['losses']
        plt.plot(losses, alpha=0.3, color='red')
        if len(losses) > window_size:
            smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(losses)), smoothed, color='red')
        plt.title('Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        # Plot dynamics losses
        plt.subplot(2, 3, 4)
        dynamics_losses = self.metrics['dynamics_losses']
        plt.plot(dynamics_losses, alpha=0.3, color='purple')
        if len(dynamics_losses) > window_size:
            smoothed = np.convolve(dynamics_losses, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(dynamics_losses)), smoothed, color='purple')
        plt.title('Dynamics Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        # Plot curriculum levels
        plt.subplot(2, 3, 5)
        curriculum = self.metrics['curriculum_levels']
        plt.plot(curriculum, color='orange', drawstyle='steps-post')
        plt.title('Curriculum Level')
        plt.xlabel('Episode')
        plt.ylabel('Level')
        plt.yticks(range(6))  # 0-5 levels
        
        # Plot recovery times
        plt.subplot(2, 3, 6)
        if 'recovery_times' in self.metrics and len(self.metrics['recovery_times']) > 0:
            recovery = self.metrics['recovery_times']
            plt.hist(recovery, bins=20, color='brown', alpha=0.7)
            plt.axvline(np.mean(recovery), color='black', linestyle='dashed', linewidth=2)
            plt.title(f'Recovery Time (Mean: {np.mean(recovery):.1f} frames)')
            plt.xlabel('Frames to Recover')
            plt.ylabel('Count')
        else:
            plt.text(0.5, 0.5, 'No recovery data yet', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Recovery Times')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.log_dir}/training_metrics.png", dpi=200)
        
        plt.close()
    
    def visualize_embeddings(self, embeddings, difficulty_changes=None, save=True):
        """
        Visualize the dynamics embeddings to detect pattern changes.
        
        Args:
            embeddings: List of dynamics embeddings from the agent
            difficulty_changes: List of frames where difficulty changed
            save: Whether to save the plot to disk
        """
        if len(embeddings) < 10:
            return
            
        # Convert list of embeddings to numpy array
        embeddings_array = np.concatenate(embeddings, axis=0)
        
        # Apply dimensionality reduction with PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings_array)
        
        # Apply t-SNE for better visualization if we have enough samples
        if len(embeddings_array) >= 50:
            try:
                tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings_array)-1), 
                          n_iter=300, random_state=42)
                tsne_result = tsne.fit_transform(embeddings_array)
            except:
                tsne_result = pca_result  # Fallback to PCA if t-SNE fails
        else:
            tsne_result = pca_result
        
        plt.figure(figsize=(16, 8))
        
        # Plot PCA
        plt.subplot(1, 2, 1)
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=range(len(pca_result)), 
                  cmap='viridis', alpha=0.7)
        
        if difficulty_changes is not None:
            # Mark the difficulty change points
            for change_idx in difficulty_changes:
                if 0 <= change_idx < len(pca_result):
                    plt.scatter(pca_result[change_idx, 0], pca_result[change_idx, 1], 
                              color='red', s=100, marker='x')
        
        plt.title(f'PCA of Dynamics Embeddings (n={len(embeddings_array)})')
        plt.colorbar(label='Time step')
        
        # Plot t-SNE
        plt.subplot(1, 2, 2)
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=range(len(tsne_result)), 
                  cmap='viridis', alpha=0.7)
        
        if difficulty_changes is not None:
            # Mark the difficulty change points
            for change_idx in difficulty_changes:
                if 0 <= change_idx < len(tsne_result):
                    plt.scatter(tsne_result[change_idx, 0], tsne_result[change_idx, 1], 
                              color='red', s=100, marker='x')
        
        plt.title(f't-SNE of Dynamics Embeddings (n={len(embeddings_array)})')
        plt.colorbar(label='Time step')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.log_dir}/dynamics_embeddings.png", dpi=200)
        
        plt.close()
    
    def plot_performance_by_difficulty(self, save=True):
        """Plot performance metrics grouped by difficulty settings"""
        if not self.rewards_by_difficulty:
            return
        
        # Calculate mean reward for each difficulty
        difficulty_labels = []
        mean_rewards = []
        std_rewards = []
        
        for key, rewards in self.rewards_by_difficulty.items():
            if len(rewards) < 5:  # Skip difficulties with too few samples
                continue
            difficulty_labels.append(key)
            mean_rewards.append(np.mean(rewards))
            std_rewards.append(np.std(rewards))
        
        # Sort by mean reward
        sorted_indices = np.argsort(mean_rewards)
        difficulty_labels = [difficulty_labels[i] for i in sorted_indices]
        mean_rewards = [mean_rewards[i] for i in sorted_indices]
        std_rewards = [std_rewards[i] for i in sorted_indices]
        
        plt.figure(figsize=(14, 8))
        plt.barh(range(len(mean_rewards)), mean_rewards, xerr=std_rewards,
                alpha=0.7, capsize=5)
        plt.yticks(range(len(difficulty_labels)), difficulty_labels)
        plt.title('Performance by Difficulty Setting')
        plt.xlabel('Mean Reward')
        plt.ylabel('Difficulty Settings')
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.log_dir}/performance_by_difficulty.png", dpi=200)
        
        plt.close()


class VideoRecorder:
    """
    Class for recording gameplay videos
    """
    def __init__(self, env, path, fps=30):
        """
        Initialize the video recorder.
        
        Args:
            env: The environment to record
            path: Path to save the video
            fps: Frames per second
        """
        self.env = env
        self.path = path
        self.fps = fps
        self.frames = []
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
    def capture_frame(self):
        """Capture the current frame from the environment"""
        # Get current render frame from the environment
        frame = None
        
        # For ALE/Atari environments
        if hasattr(self.env, 'ale'):
            try:
                # Use ALE's get_rgb_array method which is more reliable
                frame = self.env.ale.getScreenRGB()
            except Exception as e:
                pass
                
        # If frame is still None, try the regular gym approaches
        if frame is None:
            try:
                # For gym 0.26+ where render() returns the frame directly
                frame = self.env.render()
            except Exception:
                pass
                
        # If we still don't have a frame, try with render_mode parameter
        if frame is None:
            try:
                # For gym 0.26+ with explicit render_mode
                frame = self.env.render(render_mode='rgb_array')
            except Exception:
                # Silence errors here as we'll try the next method
                pass
                
        # If still no frame, try legacy mode parameter (but suppress error message)
        if frame is None:
            try:
                frame = self.env.render(mode='rgb_array')
            except Exception:
                # Generate a blank frame if all rendering methods fail
                if self.frames and len(self.frames) > 0:
                    # Use the shape of previous frames
                    shape = self.frames[0].shape
                    frame = np.zeros(shape, dtype=np.uint8)
                else:
                    # Default size for Atari if we don't have any frames yet
                    frame = np.zeros((210, 160, 3), dtype=np.uint8)
        
        self.frames.append(frame)
    
    def save_video(self):
        """Save the recorded frames as a video"""
        if not self.frames:
            print("No frames to save")
            return
            
        # Get dimensions from the first frame
        height, width, _ = self.frames[0].shape
        
        # Create the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(self.path, fourcc, self.fps, (width, height))
        
        # Write each frame
        for frame in self.frames:
            # Convert from RGB to BGR (OpenCV uses BGR)
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Release the video writer
        video.release()
        print(f"Video saved to {self.path}")
        
        # Clear frames
        self.frames = []
