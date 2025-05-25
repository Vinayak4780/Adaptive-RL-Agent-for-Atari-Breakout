import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import time

# Define a named tuple for storing experiences in the replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Atari games
    """
    def __init__(self, input_shape, n_actions):
        """
        Initialize the DQN network.
        
        Args:
            input_shape: Shape of input state (C, H, W)
            n_actions: Number of possible actions
        """
        super(DQNNetwork, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of the flattened conv output
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Add state dynamics detection layer
        self.dynamics_layers = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Main Q-value prediction layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
          # Additional predictive layer for detecting environmental changes
        self.dynamics_predictor = nn.Linear(128, 4)  # Predict 4 environmental parameters
    
    def _get_conv_output_size(self, shape):
        """Calculate the output size of the convolutional layers"""
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """Forward pass through the network"""
        # Ensure proper input shape (batch_size, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # For PyTorch channel-first format, permute if input is channel-last
        if x.shape[1] != self.conv_layers[0].in_channels:
            # Assume input is (batch, height, width, channels)
            x = x.permute(0, 3, 1, 2)
        
        # Convolutional feature extraction
        conv_features = self.conv_layers(x)
        # Use reshape instead of view to handle non-contiguous tensors
        flattened = conv_features.reshape(conv_features.size(0), -1)
        
        # Get Q-values
        q_values = self.fc_layers(flattened)
        
        # Get dynamics embedding for environmental change detection
        dynamics_embedding = self.dynamics_layers(flattened)
        dynamics_prediction = self.dynamics_predictor(dynamics_embedding)
        
        return q_values, dynamics_prediction, dynamics_embedding

class ExperienceReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions
    """
    def __init__(self, capacity):
        """
        Initialize the replay buffer with fixed capacity.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer"""
        experiences = random.sample(self.buffer, batch_size)
        
        # Convert to tensors
        states = torch.from_numpy(np.array([e.state for e in experiences])).float()
        actions = torch.tensor([e.action for e in experiences]).long()
        rewards = torch.tensor([e.reward for e in experiences]).float()
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences])).float()
        dones = torch.tensor([e.done for e in experiences]).float()
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)

class PrioritizedExperienceReplayBuffer:
    """
    Prioritized Experience Replay Buffer for storing transitions with priorities
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Exponent determining how much prioritization is used (0=no prioritization, 1=full)
            beta: Importance sampling correction exponent
            beta_increment: Amount to increase beta each time we sample
        """
        self.buffer = []
        self.capacity = capacity
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer with maximum priority"""
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # New experiences get maximum priority
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample a batch of experiences based on priorities"""
        if len(self.buffer) < self.capacity:
            probs = self.priorities[:len(self.buffer)]
        else:
            probs = self.priorities
            
        # Convert priorities to probabilities
        probs = probs ** self.alpha
        probs = probs / np.sum(probs)
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # Normalize
        weights = torch.tensor(weights).float()
        
        # Increase beta for future sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Convert to tensors
        states = torch.from_numpy(np.array([e.state for e in experiences])).float()
        actions = torch.tensor([e.action for e in experiences]).long()
        rewards = torch.tensor([e.reward for e in experiences]).float()
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences])).float()
        dones = torch.tensor([e.done for e in experiences]).float()
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + 1e-5  # Add small constant for stability
        
        self.max_priority = max(self.max_priority, np.max(self.priorities))
    
    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)

class AdaptiveDQNAgent:
    """
    DQN Agent with adaptive features to handle dynamic environments
    """
    def __init__(
        self,
        state_shape,
        n_actions,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=10000,
        buffer_size=100000,
        batch_size=32,
        target_update_freq=1000,
        use_prioritized_replay=True,
        device=None
    ):
        """
        Initialize the adaptive DQN agent.
        
        Args:
            state_shape: Shape of the input state (channels, height, width)
            n_actions: Number of possible actions
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration probability
            epsilon_final: Final exploration probability
            epsilon_decay: Number of steps to decay epsilon over
            buffer_size: Size of the replay buffer
            batch_size: Number of samples to draw from the buffer for each update
            target_update_freq: How often to update the target network
            use_prioritized_replay: Whether to use prioritized replay
            device: Device to run the model on (cpu/cuda)
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_prioritized_replay = use_prioritized_replay
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_shape, n_actions).to(self.device)
        self.target_net = DQNNetwork(state_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is used for evaluation only
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedExperienceReplayBuffer(buffer_size)
        else:
            self.replay_buffer = ExperienceReplayBuffer(buffer_size)
        
        # Training metrics
        self.steps_done = 0
        self.episode_rewards = []
        self.avg_q_values = []
        self.losses = []
        self.dynamics_losses = []
        
        # Environmental change detection
        self.env_change_detected = False
        self.env_change_history = []
        self.embeddings_history = []
        
    def select_action(self, state, eval_mode=False):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: The current state
            eval_mode: If True, use greedy policy (epsilon=0)
        
        Returns:
            The selected action
        """
        # Calculate epsilon based on decay schedule
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  np.exp(-self.steps_done / self.epsilon_decay)
        
        if eval_mode:
            epsilon = 0.05  # Small epsilon for evaluation to ensure some exploration
        
        self.steps_done += 1
        
        # With probability epsilon, select a random action
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        
        # Otherwise, select the action with highest Q-value
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            q_values, dynamics_pred, dynamics_embedding = self.policy_net(state_tensor)
            
            # Store embedding for analysis
            if len(self.embeddings_history) < 1000:  # Limit size to avoid memory issues
                self.embeddings_history.append(dynamics_embedding.cpu().numpy())
            
            return q_values.max(1)[1].item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer"""
        # Convert states shape from (H, W, C) to (C, H, W) for PyTorch
        state_reshaped = np.transpose(state, (2, 0, 1))
        next_state_reshaped = np.transpose(next_state, (2, 0, 1))
        
        # Add to replay buffer
        self.replay_buffer.add(state_reshaped, action, reward, next_state_reshaped, done)
    
    def optimize_model(self):
        """Perform one step of optimization on the DQN"""
        # Skip if we don't have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        # Sample from replay buffer
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Compute current Q values
        current_q_values, dynamics_preds, _ = self.policy_net(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values, _, _ = self.target_net(next_states)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute TD error
        td_error = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        
        # Update priorities in prioritized replay buffer
        if self.use_prioritized_replay:
            self.replay_buffer.update_priorities(indices, td_error)
        
        # Compute Huber loss (less sensitive to outliers)
        q_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
        q_loss = (q_loss * weights).mean()
        
        # Generate target for dynamics prediction (placeholder - in real env we would extract from env)
        # Here we're just using a simple dummy target - in practice this would come from the env
        dynamics_targets = torch.zeros(self.batch_size, 4).to(self.device)
        dynamics_loss = F.mse_loss(dynamics_preds, dynamics_targets)
        
        # Combined loss with smaller weight for dynamics loss
        total_loss = q_loss + 0.1 * dynamics_loss
        
        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Record metrics
        self.losses.append(q_loss.item())
        self.dynamics_losses.append(dynamics_loss.item())
        
        return q_loss.item(), dynamics_loss.item()
    
    def update_target_network(self):
        """Update the target network with the policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def detect_environment_change(self, state, reward, next_state):
        """
        Detect if the environment dynamics have changed.
        This function uses the dynamics embedding to detect changes.
        
        Returns True if a change is detected, False otherwise.
        """
        # Convert state shape for the model
        state_tensor = torch.from_numpy(np.transpose(state, (2, 0, 1))).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, dynamics_pred, dynamics_embedding = self.policy_net(state_tensor)
            
        # Here we would implement change detection logic
        # For now, using a placeholder implementation
        
        # In a real implementation, we would:
        # 1. Keep a history of embeddings
        # 2. Compute running statistics (mean, variance)
        # 3. Detect outliers/shifts using statistical methods
        
        # Placeholder - random detection with 0.001 probability
        env_change = random.random() < 0.001
        
        if env_change:
            self.env_change_detected = True
            self.env_change_history.append(self.steps_done)
            
        return env_change
    
    def save_model(self, path):
        """Save the model to disk"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'losses': self.losses,
            'dynamics_losses': self.dynamics_losses,
            'env_change_history': self.env_change_history
        }, path)
    
    def load_model(self, path):
        """Load the model from disk"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.losses = checkpoint['losses']
        self.dynamics_losses = checkpoint['dynamics_losses']
        self.env_change_history = checkpoint['env_change_history']
