"""
PPO Policy for MyoChallenge Table Tennis Task
Loads trained PPO model and performs inference
"""

import pickle
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import tensorflow_probability.substrates.jax as tfp
from SimBa import SimBaEncoder

tfd = tfp.distributions

def layer_init(scale: float = jnp.sqrt(2)):
    """Initialize layer with orthogonal weights and constant bias."""
    return nnx.initializers.orthogonal(scale)

class RunningMeanStd:
    """Running mean and std for observation normalization"""
    
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.epsilon = epsilon

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)




class Agent(nnx.Module):
    def __init__(
        self,
        envs,
        rngs: nnx.Rngs,
        critic_hidden_dim: int = 1024,
        critic_num_blocks: int = 2,
        critic_block_type: str = "residual",
        actor_hidden_dim: int = 512,
        actor_num_blocks: int = 1,
        actor_block_type: str = "residual"
    ):
        """
        Initialize Agent with SimBa architecture.

        Args:
            envs: Vectorized gym environments
            rngs: Random number generator state
            critic_hidden_dim: Hidden dimension for critic network
            critic_num_blocks: Number of blocks in critic network
            critic_block_type: Type of blocks for critic ("residual" or "mlp")
            actor_hidden_dim: Hidden dimension for actor network
            actor_num_blocks: Number of blocks in actor network
            actor_block_type: Type of blocks for actor ("residual" or "mlp")
        """
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        action_shape = envs.single_action_space.shape

        obs_dim = obs_shape[0]
        action_dim = action_shape[0]

        # Critic network: SimBa encoder + output layer
        self.critic_encoder = SimBaEncoder(
            input_dim=obs_dim,
            hidden_dim=critic_hidden_dim,
            num_blocks=critic_num_blocks,
            block_type=critic_block_type,
            rngs=rngs
        )
        self.critic_head = nnx.Linear(
            critic_hidden_dim,
            1,
            rngs=rngs,
            kernel_init=layer_init(1.0),
            bias_init=nnx.initializers.constant(0.0)
        )

        # Actor network: SimBa encoder + output layer
        self.actor_encoder = SimBaEncoder(
            input_dim=obs_dim,
            hidden_dim=actor_hidden_dim,
            num_blocks=actor_num_blocks,
            block_type=actor_block_type,
            rngs=rngs
        )
        self.actor_mean_head = nnx.Linear(
            actor_hidden_dim,
            action_dim,
            rngs=rngs,
            kernel_init=layer_init(0.01),
            bias_init=nnx.initializers.constant(0.0)
        )

        # Actor log std parameter - initialize as vector for proper broadcasting
        self.actor_logstd = nnx.Param(jnp.zeros((1, action_dim)))

    def get_value(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Get state value from the critic network.

        Args:
            obs: Normalized observation tensor of shape (batch_size, obs_dim)

        Returns:
            State value tensor of shape (batch_size, 1)
        """
        x = self.critic_encoder(obs)
        return self.critic_head(x)

    def get_action_and_value(self, obs: jnp.ndarray, key: jax.random.PRNGKey, action: jnp.ndarray = None):
        """
        Get action, log probability, entropy, and value from the network.

        Args:
            obs: Normalized observation tensor of shape (batch_size, obs_dim)
            key: JAX random key for sampling
            action: Optional action tensor for computing log prob of specific actions

        Returns:
            Tuple of (action, log_prob, entropy, value)
            - action: shape (batch_size, action_dim)
            - log_prob: shape (batch_size,)
            - entropy: shape (batch_size,)
            - value: shape (batch_size, 1)
        """
        # Get action mean from actor network
        x = self.actor_encoder(obs)
        action_mean = self.actor_mean_head(x)

        # Expand log std to match action mean shape
        action_logstd = jnp.broadcast_to(self.actor_logstd.value, action_mean.shape)
        action_std = jnp.exp(action_logstd)

        # Create normal distribution
        probs = tfd.Normal(action_mean, action_std)

        # Sample action if not provided
        if action is None:
            action = probs.sample(seed=key)

        # Compute log probability and entropy (sum over action dimensions)
        log_prob = jnp.sum(probs.log_prob(action), axis=1)
        entropy = jnp.sum(probs.entropy(), axis=1)

        # Get state value
        value = self.get_value(obs)

        return action, log_prob, entropy, value

class PPOAgent():
    """Wrapper class for PPO policy inference"""

    def __init__(self, model_path='model_step_4862752.pkl'):
        """
        Initialize PPO agent for inference.

        Args:
            env: Gym environment (used to get observation/action space)
            model_path: Path to trained model checkpoint
        """

        # Initialize observation normalization
        self.obs_rms = RunningMeanStd(shape=(417,))

        # Create a dummy environment wrapper for Agent initialization
        class DummyEnv:
            def __init__(self, obs_dim, action_dim):
                self.single_observation_space = type('obj', (object,), {'shape': (obs_dim,)})()
                self.single_action_space = type('obj', (object,), {'shape': (action_dim,)})()

        dummy_env = DummyEnv(427, 275)

        # Initialize agent with same architecture as training
        rngs = nnx.Rngs(0)
        self.agent = Agent(
            envs=dummy_env,
            rngs=rngs,
            critic_hidden_dim=1024,
            critic_num_blocks=2,
            critic_block_type='residual',
            actor_hidden_dim=512,
            actor_num_blocks=1,
            actor_block_type='residual'
        )

        # Load trained model
        self.load_model(model_path)

        print(f"✅ PPO Policy loaded successfully")
        print(f"   Model: {model_path}")

    def load_model(self, checkpoint_path):
        """Load model weights and normalization stats"""
        with open(checkpoint_path, 'rb') as f:
            save_data = pickle.load(f)

        # Load model state
        model_state = save_data['model_state']
        nnx.update(self.agent, model_state)

        # Load normalization statistics
        if 'norm_stats' in save_data:
            norm_stats = save_data['norm_stats']
            self.obs_rms.mean = norm_stats['mean']
            self.obs_rms.var = norm_stats['var']
            self.obs_rms.count = norm_stats['count']
            if 'epsilon' in norm_stats:
                self.obs_rms.epsilon = norm_stats['epsilon']
            print(f"   Loaded normalization stats")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Get action for given observation (deterministic for evaluation)"""
        # Normalize observation
        obs_normalized = self.obs_rms.normalize(obs)

        # Convert to JAX array and add batch dimension
        obs_jax = jnp.array(obs_normalized).reshape(1, -1)

        # Get action mean from actor (deterministic policy for evaluation)
        x = self.agent.actor_encoder(obs_jax)
        action_mean = self.agent.actor_mean_head(x)

        # Convert back to numpy and remove batch dimension
        action_np = np.array(action_mean).flatten()


        return action_np

