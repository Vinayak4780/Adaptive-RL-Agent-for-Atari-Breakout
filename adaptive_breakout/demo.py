"""
This script demonstrates the adaptive RL agent on Atari Breakout
with a small number of episodes for quick validation.
"""

import os
import sys
import time
from datetime import datetime

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from adaptive_breakout.environment.dynamic_breakout import make_dynamic_breakout
from adaptive_breakout.environment.preprocessing import make_preprocessed_env
from adaptive_breakout.agent.dqn_agent import AdaptiveDQNAgent
from adaptive_breakout.utils.metrics import VideoRecorder

def main():
    """Run a demonstration of the adaptive agent"""
    print("Initializing Dynamic Breakout environment...")
    
    # Create environment
    env = make_dynamic_breakout(curriculum_level=0)
    env = make_preprocessed_env(env)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("demo_results", f"demo_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
    
    # Initialize agent
    state_shape = (4, 84, 84)  # (C, H, W) for PyTorch
    n_actions = env.action_space.n
    
    print(f"Creating DQN agent with {n_actions} actions...")
    agent = AdaptiveDQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        buffer_size=10000,  # Smaller buffer for demo
        batch_size=32,
    )
    
    # Run a short training loop
    num_episodes = 5
    print(f"Running {num_episodes} training episodes...")
    
    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Set up video recorder for first and last episodes
        if episode == 0 or episode == num_episodes - 1:
            video_path = os.path.join(output_dir, "videos", f"episode_{episode+1}.mp4")
            recorder = VideoRecorder(env, video_path)
        else:
            recorder = None
            
        while not done:
            # Record frame if applicable
            if recorder:
                recorder.capture_frame()
                
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update the agent
            if len(agent.replay_buffer) > agent.batch_size:
                agent.optimize_model()
                
            # Update state and counters
            state = next_state
            total_reward += reward
            steps += 1
            
            # Limit demo length
            if steps >= 1000:
                break
        
        # Save video if applicable
        if recorder:
            recorder.save_video()
            print(f"Video saved to {video_path}")
            
        print(f"Episode {episode+1} - Reward: {total_reward}, Steps: {steps}")
        
    # Save model
    model_path = os.path.join(output_dir, "agent_demo.pth")
    agent.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Test with different curriculum levels
    print("\nTesting agent with different curriculum levels...")
    
    for level in range(3):  # Test the first few levels
        print(f"Curriculum level {level}")
        
        # Create new environment with this curriculum level
        env.close()
        env = make_dynamic_breakout(curriculum_level=level)
        env = make_preprocessed_env(env)
        
        # Record a video
        video_path = os.path.join(output_dir, "videos", f"curriculum_level_{level}.mp4")
        recorder = VideoRecorder(env, video_path)
        
        # Run a test episode
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Record frame
            recorder.capture_frame()
            
            # Select action
            action = agent.select_action(state, eval_mode=True)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update state and counters
            state = next_state
            total_reward += reward
            steps += 1
            
            # Limit demo length
            if steps >= 500:
                break
        
        # Save video
        recorder.save_video()
        print(f"Curriculum level {level} - Reward: {total_reward}, Steps: {steps}")
        print(f"Video saved to {video_path}")
    
    # Clean up
    env.close()
    print("\nDemo complete!")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
