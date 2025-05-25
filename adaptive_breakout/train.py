import os
import numpy as np
import torch
import random
import argparse
import time
from datetime import datetime

# Import our custom modules
from adaptive_breakout.environment.dynamic_breakout import make_dynamic_breakout
from adaptive_breakout.environment.preprocessing import make_preprocessed_env
from adaptive_breakout.agent.dqn_agent import AdaptiveDQNAgent
from adaptive_breakout.utils.metrics import MetricsTracker, VideoRecorder
from adaptive_breakout.utils.curriculum import CurriculumScheduler

def set_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def create_env(curriculum_level=0, seed=None):
    """Create and wrap the environment"""
    # Create the dynamic Breakout environment
    env = make_dynamic_breakout(curriculum_level=curriculum_level, seed=seed)
    
    # Apply preprocessing
    env = make_preprocessed_env(env, frame_skip=4, frame_size=84, stack_frames=4)
    
    return env

def train(args):
    """Main training function"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"dqn_adaptive_breakout_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seeds for reproducibility
    set_seeds(args.seed)
    
    # Create environment
    env = create_env(curriculum_level=0, seed=args.seed)
    
    # Get environment specs
    state_shape = (4, 84, 84)  # (C, H, W) for PyTorch
    n_actions = env.action_space.n
    
    # Initialize agent
    agent = AdaptiveDQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        use_prioritized_replay=args.prioritized_replay
    )
    
    # Initialize curriculum scheduler
    curriculum = CurriculumScheduler(
        n_levels=6,
        auto_curriculum=args.auto_curriculum,
        performance_window=10
    )
    
    # Initialize metrics tracker
    metrics = MetricsTracker(os.path.join(output_dir, "logs"))
    
    # Initialize video recorder
    video_recorder = VideoRecorder(
        env, 
        os.path.join(output_dir, "videos", "training_episode_0.mp4")
    )
    
    # Training loop
    total_steps = 0
    episodes_since_eval = 0
    episodes_since_video = 0
    
    for episode in range(args.num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_q_values = []
        episode_losses = []
        episode_dynamics_losses = []
        difficulty_changes = []
        
        # Check if curriculum level should change
        if episode > 0 and curriculum.update(episode_reward, episode):
            # Curriculum level changed, update environment
            new_level = curriculum.get_level()
            print(f"Curriculum level increased to {new_level}")
            
            # Create new environment with updated curriculum level
            env.close()
            env = create_env(curriculum_level=new_level, seed=args.seed)
            
            # Record a video after curriculum change
            episodes_since_video = args.video_freq  # Force a video recording
        
        # Record video for some episodes
        record_video = episodes_since_video >= args.video_freq
        if record_video:
            video_path = os.path.join(output_dir, "videos", f"training_episode_{episode}.mp4")
            video_recorder = VideoRecorder(env, video_path)
            episodes_since_video = 0
        else:
            episodes_since_video += 1
            
        # Episode loop
        recovery_start = None
        while not done:
            # Capture video frame if recording
            if record_video:
                video_recorder.capture_frame()
                
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Detect environment changes
            if args.detect_changes and agent.detect_environment_change(state, reward, next_state):
                difficulty_changes.append(episode_length)
                
                # Start tracking recovery time
                if recovery_start is None:
                    recovery_start = episode_length
            
            # Update the agent
            if total_steps % args.update_freq == 0 and total_steps > args.learning_starts:
                loss, dynamics_loss = agent.optimize_model()
                episode_losses.append(loss)
                episode_dynamics_losses.append(dynamics_loss)
            
            # Update the target network
            if total_steps % args.target_update_freq == 0 and total_steps > args.learning_starts:
                agent.update_target_network()
            
            # Track Q-values for monitoring
            if total_steps % 10 == 0:  # Only track every few steps to save computation
                with torch.no_grad():
                    state_tensor = torch.from_numpy(np.transpose(state, (2, 0, 1))).float().unsqueeze(0).to(agent.device)
                    q_values, _, _ = agent.policy_net(state_tensor)
                    episode_q_values.append(q_values.max().item())
            
            # Update state, reward, and counters
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # If we detected a change and now performance is recovering, log the recovery time
            if recovery_start is not None and reward > 0:
                recovery_time = episode_length - recovery_start
                metrics.log_recovery_time(recovery_time)
                recovery_start = None  # Reset for next change
            
        # End of episode
        
        # If recording, save the video
        if record_video:
            video_recorder.save_video()
        
        # Calculate episode metrics
        mean_q = np.mean(episode_q_values) if episode_q_values else 0
        mean_loss = np.mean(episode_losses) if episode_losses else 0
        mean_dynamics_loss = np.mean(episode_dynamics_losses) if episode_dynamics_losses else 0
        
        # Calculate epsilon for this episode
        epsilon = agent.epsilon_final + (agent.epsilon_start - agent.epsilon_final) * \
                 np.exp(-agent.steps_done / agent.epsilon_decay)
        
        # Log episode metrics
        metrics.log_episode(
            episode_idx=episode,
            episode_reward=episode_reward,
            episode_length=episode_length,
            mean_q_value=mean_q,
            loss=mean_loss,
            dynamics_loss=mean_dynamics_loss,
            epsilon=epsilon,
            curriculum_level=curriculum.get_level(),
            env_changes=len(difficulty_changes)
        )
        
        # Plot metrics periodically
        if episode % args.plot_freq == 0:
            metrics.plot_metrics()
            
            # Plot dynamics embeddings if we have enough data
            if agent.embeddings_history:
                metrics.visualize_embeddings(
                    agent.embeddings_history, 
                    difficulty_changes=agent.env_change_history
                )
                
            metrics.plot_performance_by_difficulty()
        
        # Save model periodically
        if episode % args.save_freq == 0 and episode > 0:
            agent.save_model(os.path.join(output_dir, f"agent_episode_{episode}.pth"))
        
        # Run evaluation episodes
        episodes_since_eval += 1
        if episodes_since_eval >= args.eval_freq:
            evaluate(agent, curriculum.get_level(), args.eval_episodes, metrics, output_dir, episode)
            episodes_since_eval = 0
    
    # Final save
    agent.save_model(os.path.join(output_dir, "agent_final.pth"))
    
    # Final metrics
    metrics.plot_metrics()
    metrics.visualize_embeddings(
        agent.embeddings_history, 
        difficulty_changes=agent.env_change_history
    )
    metrics.plot_performance_by_difficulty()
    
    # Close environment
    env.close()

def evaluate(agent, curriculum_level, n_episodes, metrics, output_dir, training_episode):
    """Evaluate the agent's performance"""
    print(f"Evaluating agent at curriculum level {curriculum_level}...")
    
    # Create environment with current curriculum level
    env = create_env(curriculum_level=curriculum_level)
    
    # Create video recorder for the first evaluation episode
    video_path = os.path.join(output_dir, "videos", f"eval_episode_{training_episode}.mp4")
    video_recorder = VideoRecorder(env, video_path)
    
    total_rewards = []
    
    for i in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        record_video = (i == 0)  # Record only first episode
        
        while not done:
            if record_video:
                video_recorder.capture_frame()
                
            # Select action (with minimal exploration)
            action = agent.select_action(state, eval_mode=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
        # End of episode
        total_rewards.append(episode_reward)
        
        if record_video:
            video_recorder.save_video()
    
    # Calculate evaluation metrics
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"Evaluation results: mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Log evaluation metrics
    metrics.add_metric('eval_rewards', mean_reward)
    metrics.add_metric('eval_std_rewards', std_reward)
    
    # Close environment
    env.close()
    
    return mean_reward, std_reward

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train adaptive RL agent for dynamic Breakout")
    
    # Environment parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Training parameters
    parser.add_argument("--num-episodes", type=int, default=10000, help="Number of episodes to train for")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Starting epsilon for exploration")
    parser.add_argument("--epsilon-final", type=float, default=0.01, help="Final epsilon for exploration")
    parser.add_argument("--epsilon-decay", type=int, default=100000, help="Decay rate for epsilon")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Size of replay buffer")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--target-update-freq", type=int, default=1000, help="Target network update frequency")
    parser.add_argument("--update-freq", type=int, default=4, help="Model update frequency")
    parser.add_argument("--learning-starts", type=int, default=10000, help="Steps before starting training")
    parser.add_argument("--prioritized-replay", action="store_true", help="Use prioritized experience replay")
    
    # Curriculum parameters
    parser.add_argument("--auto-curriculum", action="store_true", help="Enable automatic curriculum")
    parser.add_argument("--detect-changes", action="store_true", help="Enable environment change detection")
    
    # Evaluation and logging parameters
    parser.add_argument("--eval-freq", type=int, default=50, help="Evaluation frequency in episodes")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--video-freq", type=int, default=50, help="Video recording frequency in episodes")
    parser.add_argument("--plot-freq", type=int, default=10, help="Plot frequency in episodes")
    parser.add_argument("--save-freq", type=int, default=100, help="Model save frequency in episodes")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Set default values for auto_curriculum and detect_changes
    if not args.auto_curriculum:
        args.auto_curriculum = True
    if not args.detect_changes:
        args.detect_changes = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train(args)

if __name__ == "__main__":
    main()
