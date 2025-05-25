import numpy as np
import time

class CurriculumScheduler:
    """
    Scheduler for curriculum learning that adjusts difficulty based on agent performance
    """
    def __init__(
        self,
        n_levels=6,  # 0-5 levels
        reward_threshold=None,
        episode_threshold=None,
        auto_curriculum=True,
        performance_window=10
    ):
        """
        Initialize the curriculum scheduler.
        
        Args:
            n_levels: Number of curriculum levels (including level 0)
            reward_threshold: Reward threshold to increase difficulty (per level)
            episode_threshold: Episode threshold to increase difficulty
            auto_curriculum: Whether to automatically adjust curriculum level
            performance_window: Number of episodes to average performance over
        """
        self.n_levels = n_levels
        self.current_level = 0
        self.auto_curriculum = auto_curriculum
        self.performance_window = performance_window
        
        # If reward thresholds not provided, use default values based on Breakout
        # These are approx. values that might need adjustment
        if reward_threshold is None:
            self.reward_thresholds = {
                0: 5,    # Standard Breakout - just hit a few bricks
                1: 10,   # Paddle speed variations - hit more bricks
                2: 15,   # Ball speed changes - break through first layer
                3: 20,   # Brick regeneration - survive longer
                4: 25    # All modifications - demonstrate mastery
            }
        else:
            self.reward_thresholds = reward_threshold
        
        # If episode thresholds not provided, use default values
        if episode_threshold is None:
            self.episode_thresholds = {
                0: 50,
                1: 100,
                2: 150,
                3: 200,
                4: 250
            }
        else:
            self.episode_thresholds = episode_threshold
            
        # Performance tracking
        self.episode_count = 0
        self.recent_rewards = []
        self.level_start_episode = 0
        self.level_start_time = time.time()
    
    def update(self, episode_reward, episode_count):
        """
        Update the curriculum level based on agent performance.
        
        Args:
            episode_reward: Reward from the latest episode
            episode_count: Current episode count
            
        Returns:
            bool: True if the curriculum level changed, False otherwise
        """
        self.episode_count = episode_count
        self.recent_rewards.append(episode_reward)
        
        # Keep only the most recent episodes
        if len(self.recent_rewards) > self.performance_window:
            self.recent_rewards.pop(0)
        
        # Only update curriculum if auto_curriculum is enabled
        if not self.auto_curriculum:
            return False
            
        # Check if we should increase difficulty
        if self.current_level < self.n_levels - 1:
            avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0
            episodes_at_level = episode_count - self.level_start_episode
            
            # Check if we've met both reward and episode thresholds
            reward_threshold = self.reward_thresholds.get(self.current_level, float('inf'))
            episode_threshold = self.episode_thresholds.get(self.current_level, float('inf'))
            
            if (avg_reward >= reward_threshold and 
                episodes_at_level >= episode_threshold and 
                len(self.recent_rewards) >= self.performance_window):
                
                # Increase curriculum level
                self.current_level += 1
                self.level_start_episode = episode_count
                self.level_start_time = time.time()
                self.recent_rewards = []
                return True
                
        return False
    
    def set_level(self, level):
        """Manually set the curriculum level"""
        assert 0 <= level < self.n_levels, f"Level must be between 0 and {self.n_levels-1}"
        if level != self.current_level:
            self.current_level = level
            self.level_start_episode = self.episode_count
            self.level_start_time = time.time()
            self.recent_rewards = []
            return True
        return False
    
    def get_level(self):
        """Get the current curriculum level"""
        return self.current_level
    
    def get_level_info(self):
        """Get information about the current curriculum level"""
        episodes_at_level = self.episode_count - self.level_start_episode
        time_at_level = time.time() - self.level_start_time
        
        return {
            'level': self.current_level,
            'episodes_at_level': episodes_at_level,
            'time_at_level': time_at_level,
            'avg_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0
        }
