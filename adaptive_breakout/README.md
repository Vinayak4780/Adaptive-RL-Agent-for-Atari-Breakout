# Adaptive RL Agent for Atari Breakout with Dynamic Difficulty

This project implements a Deep Q-Network (DQN) agent that can play Atari Breakout while adapting to dynamically changing game conditions during gameplay.

## Features

- Custom dynamic Breakout environment with:
  - Variable paddle speed (50% to 150% of normal)
  - Dynamic ball speed changes
  - Brick regeneration
  - Paddle size changes
- DQN implementation with:
  - Experience replay (regular and prioritized)
  - Separate target network
  - Convolutional neural network architecture
- Adaptive mechanisms:
  - Internal representation learning for dynamics detection
  - Recovery strategies after environmental changes
- Curriculum learning:
  - Gradual introduction of difficulty changes
  - Automatic progression based on performance
- Performance analysis:
  - Visualization of agent adaptation
  - Analysis of internal representations
  - Recovery time metrics

## Project Structure

```
adaptive_breakout/
├── agent/               # Agent implementation
│   ├── dqn_agent.py     # DQN agent with adaptive features
│   └── __init__.py
├── environment/         # Custom environment
│   ├── dynamic_breakout.py  # Dynamic Breakout environment
│   ├── preprocessing.py     # Atari preprocessing wrappers
│   └── __init__.py
├── utils/               # Utility modules
│   ├── curriculum.py    # Curriculum learning scheduler
│   ├── metrics.py       # Metrics tracking and visualization
│   └── __init__.py
├── analysis/            # Analysis tools
│   ├── analyze_agent.py # Analysis of agent performance
│   └── __init__.py
├── train.py             # Main training script
└── __init__.py
```

## Requirements

- Python 3.7+
- PyTorch
- OpenAI Gym
- Atari Learning Environment (ALE)
- NumPy, Matplotlib, OpenCV

## Getting Started

Make sure you have the Atari ROM for Breakout imported using the Atari Learning Environment:

```
ale-import-roms
```

### Training the Agent

To train the agent with default parameters:

```
python -m adaptive_breakout.train --auto-curriculum --detect-changes --prioritized-replay
```

Key training parameters:
- `--num-episodes`: Number of episodes to train (default: 10000)
- `--auto-curriculum`: Enable automatic curriculum progression
- `--detect-changes`: Enable environment change detection
- `--prioritized-replay`: Use prioritized experience replay

### Analyzing the Agent

After training, analyze the agent's performance:

```
python -m adaptive_breakout.analysis.analyze_agent --model-path results/model_final.pth --log-dir results/logs --output-dir analysis_results
```

## Curriculum Levels

The agent is trained through progressive difficulty levels:

0. Standard Breakout (no modifications)
1. Paddle speed variations (50% to 150% of normal)
2. Level 1 + Ball speed changes
3. Level 2 + Brick regeneration
4. Level 3 + Paddle size changes
5. All modifications with increased frequency

## Implementation Details

### Dynamic Breakout Environment

The `DynamicBreakout` class wraps the standard Atari Breakout environment and introduces dynamic difficulty changes:
- Randomly varies paddle speed
- Increases ball speed occasionally
- Regenerates destroyed bricks
- Changes paddle size periodically

### DQN Agent

The `AdaptiveDQNAgent` class implements a DQN agent with:
- Convolutional neural network for state processing
- Additional networks for dynamics prediction
- Experience replay buffer
- Environmental change detection

### Curriculum Learning

The `CurriculumScheduler` class manages the progression of difficulty:
- Monitors agent performance
- Determines when to increase difficulty
- Provides curriculum level information

## Results and Analysis

The `analyze_agent.py` script generates comprehensive analysis including:
- Performance across curriculum levels
- Adaptation to difficulty changes
- Recovery time analysis
- Visualization of internal representations
- Comprehensive final report

## Visualizations

The project generates various visualizations:
- Training progress metrics
- Dynamics embedding visualizations
- Performance by difficulty setting
- Recovery time distributions
- Gameplay videos at different difficulty levels
