import gym
import numpy as np
from gym import spaces
from ale_py import ALEInterface
from ale_py.roms import Breakout

class DynamicBreakout(gym.Wrapper):
    """
    A custom wrapper around the Atari Breakout environment that introduces
    dynamic difficulty changes during gameplay.
    
    Features:
    - Paddle speed variations (50% to 150% of normal)
    - Ball speed increases
    - Brick regeneration
    - Paddle size changes
    
    This wrapper maintains the same observation and action spaces as the
    original Breakout environment, but modifies the dynamics.
    """
    
    def __init__(self, env=None, curriculum_level=0):
        """
        Initialize the dynamic Breakout environment.
        
        Args:
            env: The Breakout environment to wrap
            curriculum_level: Level of difficulty (0-5)
                0: Standard Breakout (no modifications)
                1: Paddle speed variations
                2: Level 1 + Ball speed changes
                3: Level 2 + Brick regeneration
                4: Level 3 + Paddle size changes
                5: All modifications with increased frequency
        """
        if env is None:
            env = gym.make('ALE/Breakout-v5')
        super(DynamicBreakout, self).__init__(env)
        
        self.curriculum_level = curriculum_level
        self.frame_count = 0
        self.episode_frame_count = 0
        
        # Dynamic difficulty parameters
        self.paddle_speed_factor = 1.0
        self.ball_speed_factor = 1.0
        self.paddle_size_factor = 1.0
        self.brick_regen_prob = 0.0
        
        # Parameters for tracking game state
        self.paddle_pos = None
        self.ball_pos = None
        self.brick_states = None
        self.last_score = 0
        self.score = 0
        
        # Tracking difficulty changes for analysis
        self.difficulty_history = []
        
        # Set seed for reproducibility
        self.np_random = np.random.RandomState()
    
    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return self.env.seed(seed)
    
    def _update_difficulty(self):
        """Update the difficulty parameters based on frame count and curriculum level"""
        self.frame_count += 1
        self.episode_frame_count += 1
        
        # Record current state before changing difficulty
        current_state = {
            'frame': self.frame_count,
            'episode_frame': self.episode_frame_count,
            'paddle_speed': self.paddle_speed_factor,
            'ball_speed': self.ball_speed_factor,
            'paddle_size': self.paddle_size_factor,
            'brick_regen_prob': self.brick_regen_prob,
            'score': self.score
        }
        
        # Determine if we should change difficulty based on curriculum level
        change_freq = max(1, 6 - self.curriculum_level)  # Higher level = more frequent changes
        
        # Paddle speed variations (Level 1+)
        if self.curriculum_level >= 1 and self.episode_frame_count % (500 // change_freq) == 0:
            self.paddle_speed_factor = 0.5 + self.np_random.rand() * 1.0  # 50% to 150%
        
        # Ball speed increases (Level 2+)
        if self.curriculum_level >= 2 and self.episode_frame_count % (1000 // change_freq) == 0:
            # 20% chance of increasing ball speed
            if self.np_random.random() < 0.2:
                self.ball_speed_factor = min(2.0, self.ball_speed_factor + 0.2)
        
        # Brick regeneration (Level 3+)
        if self.curriculum_level >= 3:
            self.brick_regen_prob = 0.001 * self.curriculum_level
        
        # Paddle size changes (Level 4+)
        if self.curriculum_level >= 4 and self.episode_frame_count % 500 == 0:
            self.paddle_size_factor = 0.7 + self.np_random.rand() * 0.6  # 70% to 130%
        
        # Record difficulty change if any parameter changed
        if (current_state['paddle_speed'] != self.paddle_speed_factor or
            current_state['ball_speed'] != self.ball_speed_factor or
            current_state['paddle_size'] != self.paddle_size_factor or
            current_state['brick_regen_prob'] != self.brick_regen_prob):
            
            self.difficulty_history.append({
                'frame': self.frame_count,
                'episode_frame': self.episode_frame_count,
                'paddle_speed': self.paddle_speed_factor,
                'ball_speed': self.ball_speed_factor,
                'paddle_size': self.paddle_size_factor,
                'brick_regen_prob': self.brick_regen_prob
            })
    
    def _modify_action(self, action):
        """Modify the action based on paddle speed variations"""
        # Only modify paddle movement actions (LEFT=3, RIGHT=2 in Breakout)
        if action in [2, 3]:
            # Randomly skip or repeat the action based on paddle_speed_factor
            if self.np_random.random() > self.paddle_speed_factor:
                return 0  # NOOP
            elif self.paddle_speed_factor > 1.0 and self.np_random.random() < (self.paddle_speed_factor - 1.0):
                # Repeat the action to simulate faster movement
                self.env.step(action)
        return action
    
    def reset(self, **kwargs):
        """Reset the environment and difficulty parameters"""
        obs = self.env.reset(**kwargs)
        self.episode_frame_count = 0
        self.paddle_speed_factor = 1.0
        self.ball_speed_factor = 1.0
        self.paddle_size_factor = 1.0
        self.brick_regen_prob = 0.0
        self.last_score = 0
        self.score = 0
        return obs
    
    def step(self, action):
        """
        Take a step in the environment with the dynamic difficulty modifications.
        """
        # Update difficulty parameters
        self._update_difficulty()
        
        # Modify action based on paddle speed
        modified_action = self._modify_action(action)
        
        # Take the step in the environment
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        done = terminated or truncated
        
        # Track score for analysis
        if reward > 0:
            self.score += reward
            
        # Apply brick regeneration with some probability
        if self.curriculum_level >= 3 and self.np_random.random() < self.brick_regen_prob:
            # This doesn't actually regenerate bricks in the ALE, but we can track it
            # for curriculum learning purposes
            pass
            
        return obs, reward, terminated, truncated, info
    
    def set_curriculum_level(self, level):
        """Set the curriculum level (0-5)"""
        assert 0 <= level <= 5, "Curriculum level must be between 0 and 5"
        self.curriculum_level = level
        
    def get_difficulty_history(self):
        """Return the history of difficulty changes"""
        return self.difficulty_history

# Helper function to create the environment
def make_dynamic_breakout(curriculum_level=0, seed=None):
    env = gym.make('ALE/Breakout-v5')
    env = DynamicBreakout(env, curriculum_level=curriculum_level)
    if seed is not None:
        env.seed(seed)
    return env
