# User Guide: Adaptive RL Agent for Atari Breakout

This guide provides detailed instructions for using our adaptive RL agent implementation for Atari Breakout with dynamic difficulty.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Running a Demo](#running-a-demo)
4. [Training the Agent](#training-the-agent)
5. [Analyzing Results](#analyzing-results)
6. [Visualizing Performance](#visualizing-performance)
7. [Understanding Curriculum Levels](#understanding-curriculum-levels)
8. [Customization Options](#customization-options)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

Ensure you have the following packages installed:
- Python 3.7+
- PyTorch
- OpenAI Gym
- Atari Learning Environment (ALE)
- NumPy, Matplotlib, OpenCV
- scikit-learn (for analysis)
- seaborn (for visualization)

All these dependencies are already installed in your virtual environment.

Make sure you have Atari ROMs imported:
```
ale-import-roms
```

## Quick Start

For a quick demonstration of the agent:

```powershell
cd "c:\Users\vinay\OneDrive\Desktop\reinforcement learning"
python -m adaptive_breakout.demo
```

This will run a short demo that trains the agent for a few episodes and tests it on different difficulty levels.

## Running a Demo

The demo script (`demo.py`) provides a quick way to visualize the agent's performance:

```powershell
cd "c:\Users\vinay\OneDrive\Desktop\reinforcement learning"
python -m adaptive_breakout.demo
```

This will:
1. Train the agent for 5 episodes
2. Save videos of the first and last episodes
3. Test the agent on different curriculum levels
4. Save videos of each curriculum level

Demo results will be saved in the `demo_results` directory.

## Training the Agent

For complete training, use the `run.py` script:

```powershell
cd "c:\Users\vinay\OneDrive\Desktop\reinforcement learning"
python -m adaptive_breakout.run train --num-episodes 1000 --auto-curriculum --detect-changes --prioritized-replay
```

### Key Training Parameters:

- `--num-episodes`: Number of episodes to train (default: 1000)
- `--learning-rate`: Learning rate for the optimizer (default: 0.0001)
- `--gamma`: Discount factor for future rewards (default: 0.99)
- `--buffer-size`: Size of the replay buffer (default: 100000)
- `--batch-size`: Batch size for training (default: 32)
- `--auto-curriculum`: Enable automatic curriculum learning
- `--detect-changes`: Enable dynamic difficulty change detection
- `--prioritized-replay`: Use prioritized experience replay
- `--output-dir`: Directory to save results (default: "results")

Training results will be saved in the specified output directory with a timestamp.

For a shorter training run suitable for testing:

```powershell
python -m adaptive_breakout.run train --num-episodes 100 --buffer-size 10000 --auto-curriculum --prioritized-replay
```

## Analyzing Results

After training, analyze the agent's performance:

```powershell
cd "c:\Users\vinay\OneDrive\Desktop\reinforcement learning"
python -m adaptive_breakout.run analyze --model-path <path_to_checkpoint> --log-dir <path_to_logs>
```

Replace `<path_to_checkpoint>` with the path to the saved model file (e.g., `results/run_20250525_123456/agent_final.pth`) and `<path_to_logs>` with the path to the logs directory (e.g., `results/run_20250525_123456/logs`).

Or, you can run both training and analysis in one command:

```powershell
python -m adaptive_breakout.run run-both --num-episodes 1000 --auto-curriculum --detect-changes --prioritized-replay
```

## Visualizing Performance

The analysis generates several visualizations:

1. **Training Metrics**: Plots of rewards, Q-values, losses, and curriculum levels over time
2. **Dynamics Embeddings**: Visualization of the agent's internal representations using PCA and t-SNE
3. **Performance by Difficulty**: Analysis of how the agent performs under different difficulty settings
4. **Recovery Times**: Distribution of how quickly the agent recovers after difficulty changes
5. **Gameplay Videos**: Recordings of the agent playing at different difficulty levels

All visualizations will be saved in the analysis output directory.

## Understanding Curriculum Levels

The adaptive agent uses curriculum learning with 6 difficulty levels:

| Level | Description | Features |
|-------|-------------|----------|
| 0 | Standard Breakout | No modifications |
| 1 | Paddle Speed Variations | Paddle speed randomly varies (50% to 150%) |
| 2 | Ball Speed Changes | Level 1 + Ball speed increases mid-episode |
| 3 | Brick Regeneration | Level 2 + Some destroyed bricks randomly reappear |
| 4 | Paddle Size Changes | Level 3 + Paddle size changes every 500 frames |
| 5 | Maximum Difficulty | All modifications with increased frequency |

With automatic curriculum learning enabled (`--auto-curriculum`), the agent will progress through these levels as its performance improves.

## Customization Options

### Modifying the Environment

To customize the dynamic difficulty parameters, edit `environment/dynamic_breakout.py`:

```python
# Examples of parameters you might want to modify:
self.paddle_speed_factor = 0.5 + self.np_random.rand() * 1.0  # 50% to 150%
self.ball_speed_factor = min(2.0, self.ball_speed_factor + 0.2)
self.brick_regen_prob = 0.001 * self.curriculum_level
```

### Customizing the Agent

To modify the agent architecture or learning parameters, edit `agent/dqn_agent.py`:

```python
# Example: Modifying the network architecture
self.conv_layers = nn.Sequential(
    nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=4, stride=2),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, stride=1),
    nn.ReLU()
)
```

### Adjusting Curriculum Learning

To change the curriculum progression criteria, edit `utils/curriculum.py`:

```python
# Example: Modifying reward thresholds for level progression
self.reward_thresholds = {
    0: 5,    # Standard Breakout
    1: 10,   # Paddle speed variations
    2: 15,   # Ball speed changes
    3: 20,   # Brick regeneration
    4: 25    # All modifications
}
```

## Troubleshooting

### Common Issues:

1. **Memory Errors**: 
   - Reduce `buffer_size` to use less memory
   - Decrease batch size
   - Use a smaller network architecture

2. **Training Too Slow**:
   - Reduce the number of episodes
   - Use a smaller replay buffer
   - Disable prioritized replay

3. **Agent Not Learning**:
   - Check learning rate (try increasing it)
   - Ensure curriculum level isn't too difficult
   - Verify environment is working correctly

4. **Video Recording Issues**:
   - Ensure you have sufficient disk space
   - Check that OpenCV is installed correctly
   - Reduce the number of frames recorded

### Debugging Tips:

- Add print statements to track the agent's progress
- Monitor Q-values to ensure they're increasing
- Check for NaN values in losses
- Visualize gameplay videos to identify issues

For additional help, refer to the code documentation within each file.
