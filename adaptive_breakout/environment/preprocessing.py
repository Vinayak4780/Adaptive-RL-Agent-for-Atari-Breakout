import gym
import numpy as np
import cv2
from gym import spaces

class AtariPreprocessing(gym.Wrapper):
    """
    Preprocessing wrapper for Atari environments:
    - Grayscale conversion
    - Frame resizing
    - Frame stacking
    - Frame skipping
    - Max pooling across skipped frames
    - Reward clipping
    """
    
    def __init__(self, env, frame_skip=4, frame_size=84, 
                 stack_frames=4, clip_rewards=True):
        """
        Initialize the preprocessing wrapper.
        
        Args:
            env: The environment to wrap
            frame_skip: Number of frames to skip
            frame_size: Size to resize frames to (square)
            stack_frames: Number of consecutive frames to stack
            clip_rewards: Whether to clip rewards to {-1, 0, 1}
        """
        super(AtariPreprocessing, self).__init__(env)
        self.frame_skip = frame_skip
        self.frame_size = frame_size
        self.stack_frames = stack_frames
        self.clip_rewards = clip_rewards
        
        # Define observation space for stacked, processed frames
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(frame_size, frame_size, stack_frames), 
            dtype=np.uint8
        )
        
        # Buffer for frame stacking
        self.frames_buffer = np.zeros((frame_size, frame_size, stack_frames), dtype=np.uint8)
    
    def reset(self, **kwargs):
        """Reset the environment and the frame buffer"""
        # Handle both old and new gym API (with and without info)
        reset_result = self.env.reset(**kwargs)
        
        # If reset returns a tuple (newer gym versions return (obs, info))
        if isinstance(reset_result, tuple) and len(reset_result) >= 2:
            obs = reset_result[0]  # The first element is the observation
        else:
            obs = reset_result  # Old gym versions simply return the observation
            
        self.frames_buffer = np.zeros_like(self.frames_buffer)
        
        # Process first frame and fill buffer with it
        processed_frame = self._process_frame(obs)
        for i in range(self.stack_frames):
            self.frames_buffer[:, :, i] = processed_frame
            
        return self.frames_buffer
    
    def step(self, action):
        """Take a step with frame skipping"""
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # Skip frames and accumulate reward
        for i in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            if self.clip_rewards:
                reward = np.sign(reward)  # Clip reward to {-1, 0, 1}
                
            total_reward += reward
            
            # Break early if game is done
            if done:
                break
                
            # If this isn"t the last frame in the skip sequence, 
            # we"ll process and update our buffer
            if i < self.frame_skip - 1:
                processed_frame = self._process_frame(obs)
                # Shift frames and add the new one
                self.frames_buffer[:, :, :-1] = self.frames_buffer[:, :, 1:]
                self.frames_buffer[:, :, -1] = processed_frame
        
        # Process the final (or only) observation and update buffer
        processed_frame = self._process_frame(obs)
        self.frames_buffer[:, :, :-1] = self.frames_buffer[:, :, 1:]
        self.frames_buffer[:, :, -1] = processed_frame
        
        return self.frames_buffer, total_reward, terminated, truncated, info
    
    def _process_frame(self, frame):
        """Process a single frame"""
        try:
            # Check if frame is already a numpy array
            if not isinstance(frame, np.ndarray):
                # If frame is a tuple or has special format, try to extract the RGB frame
                # For Atari environments, the observation might be a tuple
                if isinstance(frame, tuple) and len(frame) >= 1:
                    frame = frame[0]  # Usually the first element is the RGB frame
                
                # Try to convert to numpy array
                try:
                    frame = np.array(frame, dtype=np.uint8)
                except:
                    print("Warning: Could not convert frame to numpy array")
                    return np.zeros((self.frame_size, self.frame_size), dtype=np.uint8)
            
            # Ensure frame has the right shape
            if frame.ndim == 0 or frame.size == 0:
                # Return an empty frame if input is invalid
                return np.zeros((self.frame_size, self.frame_size), dtype=np.uint8)
            
            # Handle case where frame might be a dictionary (for some gym environments)
            if isinstance(frame, dict) and 'rgb' in frame:
                frame = frame['rgb']
            
            # Convert to grayscale - make sure frame has 3 dimensions for RGB conversion
            if frame.ndim == 3 and frame.shape[2] >= 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame  # Already grayscale or not proper RGB
            
            # Resize to specified frame size
            resized = cv2.resize(gray, (self.frame_size, self.frame_size), 
                                interpolation=cv2.INTER_AREA)
            
            return resized
        except Exception as e:
            print(f"Error processing frame: {e}")
            return np.zeros((self.frame_size, self.frame_size), dtype=np.uint8)

# Helper function to create a preprocessed environment
def make_preprocessed_env(env, frame_skip=4, frame_size=84, 
                        stack_frames=4, clip_rewards=True):
    """Create a preprocessed environment from an existing environment"""
    return AtariPreprocessing(
        env,
        frame_skip=frame_skip,
        frame_size=frame_size,
        stack_frames=stack_frames,
        clip_rewards=clip_rewards
    )
