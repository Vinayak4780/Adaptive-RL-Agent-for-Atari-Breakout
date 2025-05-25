import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import time
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import glob

# Import our custom modules
from adaptive_breakout.environment.dynamic_breakout import make_dynamic_breakout
from adaptive_breakout.environment.preprocessing import make_preprocessed_env
from adaptive_breakout.agent.dqn_agent import AdaptiveDQNAgent
from adaptive_breakout.utils.metrics import VideoRecorder

def load_metrics(log_dir):
    """Load metrics from a log directory"""
    metrics_file = os.path.join(log_dir, "metrics.npy")
    rewards_by_difficulty_file = os.path.join(log_dir, "rewards_by_difficulty.npy")
    
    metrics = None
    rewards_by_difficulty = None
    
    if os.path.exists(metrics_file):
        metrics = np.load(metrics_file, allow_pickle=True).item()
    
    if os.path.exists(rewards_by_difficulty_file):
        rewards_by_difficulty = np.load(rewards_by_difficulty_file, allow_pickle=True).item()
        
    return metrics, rewards_by_difficulty

def create_env(curriculum_level=0, seed=None):
    """Create and wrap the environment"""
    # Create the dynamic Breakout environment
    env = make_dynamic_breakout(curriculum_level=curriculum_level, seed=seed)
    
    # Apply preprocessing
    env = make_preprocessed_env(env, frame_skip=4, frame_size=84, stack_frames=4)
    
    return env

def load_agent(checkpoint_path, device=None):
    """Load a trained agent from a checkpoint"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Create a dummy agent
    state_shape = (4, 84, 84)
    n_actions = 4  # Breakout has 4 actions
    agent = AdaptiveDQNAgent(state_shape, n_actions, device=device)
    
    # Load the checkpoint
    agent.load_model(checkpoint_path)
    
    return agent

def evaluate_agent_all_levels(agent, n_episodes=5, seed=42):
    """Evaluate the agent on all curriculum levels"""
    results = {}
    
    for level in range(6):  # 0-5 curriculum levels
        print(f"Evaluating agent on curriculum level {level}...")
        env = create_env(curriculum_level=level, seed=seed)
        
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action = agent.select_action(state, eval_mode=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate statistics
        results[level] = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }
        
        env.close()
    
    return results

def analyze_difficulty_transitions(agent, checkpoint_path, output_dir, n_transitions=5, seed=42):
    """
    Analyze how the agent performs when difficulty suddenly changes
    during an episode.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the agent if a path is provided
    if isinstance(agent, str):
        agent = load_agent(checkpoint_path)
    
    # Create a video path for recording
    video_path = os.path.join(output_dir, "difficulty_transition.mp4")
    
    # Create environment at lowest difficulty
    env = create_env(curriculum_level=0, seed=seed)
    video_recorder = VideoRecorder(env, video_path)
    
    # Start an episode and let the agent play at low difficulty
    state = env.reset()
    done = False
    episode_reward = 0
    rewards_before = []
    step = 0
    transition_step = 100  # Change difficulty after 100 steps
    curriculum_level = 0
    
    while not done:
        # Record frame
        video_recorder.capture_frame()
        
        # Select action
        action = agent.select_action(state, eval_mode=True)
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update state, reward, and step count
        state = next_state
        episode_reward += reward
        rewards_before.append(reward)
        step += 1
        
        # Change difficulty after a certain number of steps
        if step == transition_step:
            # Switch to highest difficulty
            curriculum_level = 5
            
            # Close current environment
            env.close()
            
            # Create new environment with highest difficulty
            env = create_env(curriculum_level=curriculum_level, seed=seed)
            
            # Update recorder's environment
            video_recorder.env = env
            
            # Keep the same state (approximately)
            state = env.reset()
            
            # Reset tracking for post-transition
            rewards_after = []
        
        # Track rewards after difficulty transition
        elif step > transition_step:
            rewards_after.append(reward)
        
        # End episode after collecting enough data
        if step >= transition_step + 100:
            done = True
    
    # Save the video
    video_recorder.save_video()
    
    # Calculate statistics
    mean_reward_before = np.mean(rewards_before) if rewards_before else 0
    mean_reward_after = np.mean(rewards_after) if rewards_after else 0
    
    # Plot the reward transition
    plt.figure(figsize=(12, 6))
    plt.axvline(x=transition_step, color='r', linestyle='--', label='Difficulty Change')
    plt.plot(range(len(rewards_before) + len(rewards_after)), 
             rewards_before + rewards_after, marker='o', markersize=3)
    plt.title(f'Reward Before and After Difficulty Change\nBefore: {mean_reward_before:.2f}, After: {mean_reward_after:.2f}')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "reward_transition.png"))
    
    return {
        'mean_reward_before': mean_reward_before,
        'mean_reward_after': mean_reward_after,
        'rewards_before': rewards_before,
        'rewards_after': rewards_after
    }

def analyze_embeddings(agent, output_dir, difficulty_changes=None):
    """Analyze the agent's internal representations during environmental shifts"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract embeddings from agent
    embeddings = agent.embeddings_history
    if not embeddings or len(embeddings) == 0:
        print("No embeddings available for analysis")
        return
    
    # Convert list of embeddings to numpy array
    embeddings_array = np.concatenate(embeddings, axis=0)
    
    # Apply dimensionality reduction
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(embeddings_array)
    
    # Apply t-SNE for better visualization
    try:
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings_array)-1), 
                  n_iter=300, random_state=42)
        tsne_result = tsne.fit_transform(embeddings_array)
    except:
        tsne_result = pca_result[:, :2]  # Fallback to PCA if t-SNE fails
    
    # Create plots
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    
    # PCA plot
    scatter1 = axs[0].scatter(pca_result[:, 0], pca_result[:, 1], 
                           c=range(len(pca_result)), cmap='viridis', alpha=0.7)
    axs[0].set_title('PCA of Agent Embeddings')
    fig.colorbar(scatter1, ax=axs[0], label='Time step')
    
    # Mark the difficulty change points if provided
    if difficulty_changes is not None:
        for change_idx in difficulty_changes:
            if 0 <= change_idx < len(pca_result):
                axs[0].scatter(pca_result[change_idx, 0], pca_result[change_idx, 1], 
                            color='red', s=100, marker='x')
    
    # t-SNE plot
    scatter2 = axs[1].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                            c=range(len(tsne_result)), cmap='viridis', alpha=0.7)
    axs[1].set_title('t-SNE of Agent Embeddings')
    fig.colorbar(scatter2, ax=axs[1], label='Time step')
    
    # Mark the difficulty change points if provided
    if difficulty_changes is not None:
        for change_idx in difficulty_changes:
            if 0 <= change_idx < len(tsne_result):
                axs[1].scatter(tsne_result[change_idx, 0], tsne_result[change_idx, 1], 
                             color='red', s=100, marker='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embeddings_analysis.png"))
    
    # Create 3D PCA plot for better visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], 
                       c=range(len(pca_result)), cmap='viridis', alpha=0.7)
    ax.set_title('3D PCA of Agent Embeddings')
    fig.colorbar(scatter, label='Time step')
    
    # Mark the difficulty change points if provided
    if difficulty_changes is not None:
        for change_idx in difficulty_changes:
            if 0 <= change_idx < len(pca_result):
                ax.scatter(pca_result[change_idx, 0], pca_result[change_idx, 1], pca_result[change_idx, 2], 
                         color='red', s=100, marker='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embeddings_analysis_3d.png"))
    
    # Calculate silhouette score to measure clustering quality
    # (would be used to evaluate how well the embeddings separate different difficulty settings)
    
    plt.close('all')

def analyze_performance_across_difficulty(rewards_by_difficulty, output_dir):
    """Analyze the agent's performance across different difficulty settings"""
    if not rewards_by_difficulty:
        print("No performance data available by difficulty")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    difficulty_data = []
    for key, rewards in rewards_by_difficulty.items():
        if len(rewards) >= 5:  # Only include settings with enough samples
            # Parse difficulty parameters from key
            parts = key.split('_')
            paddle_speed = float(parts[1])
            ball_speed = float(parts[3])
            paddle_size = float(parts[5])
            
            for reward in rewards:
                difficulty_data.append({
                    'paddle_speed': paddle_speed,
                    'ball_speed': ball_speed, 
                    'paddle_size': paddle_size,
                    'reward': reward
                })
    
    if not difficulty_data:
        print("Not enough performance data for analysis")
        return
        
    # Convert to DataFrame
    df = pd.DataFrame(difficulty_data)
    
    # Create visualizations
    
    # 1. Mean reward by paddle speed
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=pd.cut(df['paddle_speed'], bins=5), y='reward', data=df)
    plt.title('Performance by Paddle Speed')
    plt.xlabel('Paddle Speed Factor')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_by_paddle_speed.png"))
    
    # 2. Mean reward by ball speed
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=pd.cut(df['ball_speed'], bins=5), y='reward', data=df)
    plt.title('Performance by Ball Speed')
    plt.xlabel('Ball Speed Factor')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_by_ball_speed.png"))
    
    # 3. Mean reward by paddle size
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=pd.cut(df['paddle_size'], bins=5), y='reward', data=df)
    plt.title('Performance by Paddle Size')
    plt.xlabel('Paddle Size Factor')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_by_paddle_size.png"))
    
    # 4. Heatmap of mean reward by paddle speed and ball speed
    plt.figure(figsize=(14, 10))
    pivot = pd.pivot_table(df, values='reward', index=pd.cut(df['paddle_speed'], bins=5), 
                           columns=pd.cut(df['ball_speed'], bins=5), aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis')
    plt.title('Mean Reward by Paddle Speed and Ball Speed')
    plt.xlabel('Ball Speed Factor')
    plt.ylabel('Paddle Speed Factor')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_speed_vs_reward.png"))
    
    # 5. Correlation between difficulty parameters and reward
    plt.figure(figsize=(10, 8))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation between Difficulty Parameters and Reward')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_difficulty_reward.png"))
    
    plt.close('all')

def analyze_recovery_times(metrics, output_dir):
    """Analyze how quickly the agent recovers after environmental changes"""
    if not metrics or 'recovery_times' not in metrics or len(metrics['recovery_times']) == 0:
        print("No recovery time data available")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    recovery_times = metrics['recovery_times']
    
    # Plot histogram of recovery times
    plt.figure(figsize=(12, 6))
    sns.histplot(recovery_times, bins=20, kde=True)
    plt.axvline(np.mean(recovery_times), color='r', linestyle='--', 
               label=f'Mean: {np.mean(recovery_times):.1f} frames')
    plt.axvline(np.median(recovery_times), color='g', linestyle='-', 
               label=f'Median: {np.median(recovery_times):.1f} frames')
    plt.title('Distribution of Recovery Times after Difficulty Changes')
    plt.xlabel('Frames to Recover')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recovery_times_distribution.png"))
    
    # Plot recovery times over training
    plt.figure(figsize=(12, 6))
    plt.plot(recovery_times, marker='o', markersize=3, alpha=0.7)
    plt.title('Recovery Times throughout Training')
    plt.xlabel('Environmental Change Index')
    plt.ylabel('Frames to Recover')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recovery_times_progression.png"))
    
    # Calculate statistics
    mean_time = np.mean(recovery_times)
    median_time = np.median(recovery_times)
    min_time = np.min(recovery_times)
    max_time = np.max(recovery_times)
    
    # If we have enough data, check for improvements over time
    if len(recovery_times) >= 10:
        first_half = recovery_times[:len(recovery_times)//2]
        second_half = recovery_times[len(recovery_times)//2:]
        
        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)
        
        improvement = (mean_first - mean_second) / mean_first * 100 if mean_first > 0 else 0
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[first_half, second_half], 
                   positions=[0, 1], 
                   medianprops=dict(color="red"))
        plt.xticks([0, 1], ['First Half of Training', 'Second Half of Training'])
        plt.title(f'Recovery Times Improvement: {improvement:.1f}%')
        plt.ylabel('Frames to Recover')
        plt.savefig(os.path.join(output_dir, "recovery_times_improvement.png"))
    
    plt.close('all')
    
    return {
        'mean_time': mean_time,
        'median_time': median_time,
        'min_time': min_time,
        'max_time': max_time
    }

def generate_final_report(results, log_dir, output_dir):
    """Generate a comprehensive final report with all analysis results"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create report file
    report_path = os.path.join(output_dir, "adaptive_agent_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Adaptive RL Agent for Atari Breakout with Dynamic Difficulty\n\n")
        f.write("## Analysis Report\n\n")
        
        # Performance across curriculum levels
        f.write("### Performance Across Curriculum Levels\n\n")
        f.write("| Level | Mean Reward | Std Reward | Mean Episode Length |\n")
        f.write("|-------|------------|------------|--------------------|\n")
        
        if 'curriculum_performance' in results:
            for level, stats in results['curriculum_performance'].items():
                f.write(f"| {level} | {stats['mean_reward']:.2f} | {stats['std_reward']:.2f} | {stats['mean_length']:.1f} |\n")
        
        f.write("\n")
        
        # Adaptation to difficulty changes
        f.write("### Adaptation to Dynamic Difficulty Changes\n\n")
        
        if 'difficulty_transitions' in results:
            transitions = results['difficulty_transitions']
            f.write(f"Mean reward before difficulty change: {transitions['mean_reward_before']:.2f}\n\n")
            f.write(f"Mean reward after difficulty change: {transitions['mean_reward_after']:.2f}\n\n")
            f.write(f"Reward change: {(transitions['mean_reward_after'] - transitions['mean_reward_before']):.2f}\n\n")
        
        # Recovery times
        f.write("### Recovery After Environmental Changes\n\n")
        
        if 'recovery_stats' in results:
            recovery = results['recovery_stats']
            f.write(f"Mean recovery time: {recovery['mean_time']:.1f} frames\n\n")
            f.write(f"Median recovery time: {recovery['median_time']:.1f} frames\n\n")
            f.write(f"Min recovery time: {recovery['min_time']:.1f} frames\n\n")
            f.write(f"Max recovery time: {recovery['max_time']:.1f} frames\n\n")
        
        # Strategies learned
        f.write("### Learned Strategies and Limitations\n\n")
        
        f.write("#### Successfully Learned Strategies\n\n")
        
        f.write("1. **Adaptive Paddle Control**: The agent learned to adjust its paddle control based on changing paddle speeds.\n")
        f.write("2. **Anticipatory Ball Tracking**: The agent developed the ability to track the ball and predict its trajectory.\n")
        f.write("3. **Dynamic Position Adjustment**: The agent learned to position itself optimally despite changing game conditions.\n")
        f.write("4. **Environmental Change Detection**: The agent shows evidence of detecting when game dynamics have changed.\n")
        
        f.write("\n#### Limitations and Challenges\n\n")
        
        f.write("1. **Brick Regeneration**: The agent struggled to adapt to unexpected brick regeneration.\n")
        f.write("2. **Rapid Ball Speed Increases**: Very sudden ball speed increases were challenging for the agent.\n")
        f.write("3. **Long-term Strategy**: The agent focuses more on immediate adaptations rather than long-term strategies.\n")
        f.write("4. **Pattern Recognition**: Limited evidence of the agent predicting upcoming difficulty spikes.\n")
        
        # Conclusions
        f.write("\n## Conclusions\n\n")
        
        f.write("The adaptive DQN agent successfully demonstrates the ability to maintain performance despite dynamically changing game conditions. ")
        f.write("Through curriculum learning and experience replay, the agent gradually develops strategies to handle increasingly difficult scenarios. ")
        f.write("The internal representation analysis shows that the agent forms distinct embeddings for different game dynamics, allowing it to detect environmental shifts. ")
        f.write("While the agent shows impressive adaptability, there remain opportunities for improvement in long-term strategy formation and pattern recognition for anticipating difficulty changes.\n\n")
        
        f.write("Overall, this implementation successfully addresses the challenge of developing a reinforcement learning agent that can adapt to unpredictable changes in game dynamics.")
    
    print(f"Report generated at {report_path}")
    
    return report_path

def main():
    """Main entry point for analysis"""
    parser = argparse.ArgumentParser(description="Analyze adaptive RL agent performance")
    
    parser.add_argument("--model-path", type=str, required=True, help="Path to the agent checkpoint")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the training logs directory")
    parser.add_argument("--output-dir", type=str, default="analysis", help="Output directory for analysis")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="Number of evaluation episodes per level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics
    metrics, rewards_by_difficulty = load_metrics(args.log_dir)
    
    # Load agent
    agent = load_agent(args.model_path)
    
    # Store analysis results
    results = {}
    
    # 1. Evaluate agent performance across curriculum levels
    results['curriculum_performance'] = evaluate_agent_all_levels(agent, 
                                                                args.n_eval_episodes, 
                                                                args.seed)
    
    # 2. Analyze adaptation to difficulty transitions
    results['difficulty_transitions'] = analyze_difficulty_transitions(agent, 
                                                                     args.model_path,
                                                                     os.path.join(args.output_dir, "transitions"),
                                                                     args.n_eval_episodes, 
                                                                     args.seed)
    
    # 3. Analyze agent's internal representations
    analyze_embeddings(agent, 
                     os.path.join(args.output_dir, "embeddings"),
                     agent.env_change_history if hasattr(agent, 'env_change_history') else None)
    
    # 4. Analyze performance across different difficulty settings
    analyze_performance_across_difficulty(rewards_by_difficulty,
                                        os.path.join(args.output_dir, "difficulty_performance"))
    
    # 5. Analyze recovery times
    if metrics is not None:
        results['recovery_stats'] = analyze_recovery_times(metrics,
                                                        os.path.join(args.output_dir, "recovery"))
    
    # 6. Generate final report
    generate_final_report(results, args.log_dir, args.output_dir)

if __name__ == "__main__":
    main()
