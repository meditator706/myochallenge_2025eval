"""
PPO Agent for MyoChallenge 2025 Table Tennis Task
Loads trained PPO model and performs inference in evaluation environment.
"""

import os
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from tensorflow_probability.substrates import jax as tfp

import evaluation_pb2
import evaluation_pb2_grpc
import grpc
import gymnasium as gym

from utils import RemoteConnection

tfd = tfp.distributions


# Custom observation keys (must match training configuration)
custom_obs_keys = [ 
    'pelvis_pos', 
    'body_qpos', 
    'body_qvel', 
    'ball_pos', 
    'ball_vel', 
    'paddle_pos', 
    'paddle_vel', 
    'paddle_ori', 
    'reach_err', 
    'touching_info', 
    'act',
]


def layer_init(scale: float = jnp.sqrt(2)):
    """Initialize layer with orthogonal weights and constant bias."""
    return nnx.initializers.orthogonal(scale)


class SimBaBlock(nnx.Module):
    """SimBa residual block."""
    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(
            hidden_dim, hidden_dim, rngs=rngs,
            kernel_init=layer_init(), bias_init=nnx.initializers.constant(0.0)
        )
        self.linear2 = nnx.Linear(
            hidden_dim, hidden_dim, rngs=rngs,
            kernel_init=layer_init(), bias_init=nnx.initializers.constant(0.0)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return nnx.relu(x + residual)


class SimBaEncoder(nnx.Module):
    """SimBa encoder with residual blocks."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        block_type: str,
        rngs: nnx.Rngs
    ):
        super().__init__()
        self.input_layer = nnx.Linear(
            input_dim, hidden_dim, rngs=rngs,
            kernel_init=layer_init(), bias_init=nnx.initializers.constant(0.0)
        )
        
        if block_type == "residual":
            self.blocks = [SimBaBlock(hidden_dim, rngs) for _ in range(num_blocks)]
        else:
            raise ValueError(f"Unsupported block type: {block_type}")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.relu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        return x


class RunningMeanStd:
    """Running mean and std for observation normalization."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.epsilon = epsilon

    def normalize(self, x):
        """Normalize observation using running statistics."""
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)


class Agent(nnx.Module):
    """PPO Agent with SimBa architecture."""
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        rngs: nnx.Rngs,
        critic_hidden_dim: int = 512,
        critic_num_blocks: int = 2,
        critic_block_type: str = "residual",
        actor_hidden_dim: int = 256,
        actor_num_blocks: int = 1,
        actor_block_type: str = "residual"
    ):
        super().__init__()
        
        # Critic network
        self.critic_encoder = SimBaEncoder(
            input_dim=obs_dim,
            hidden_dim=critic_hidden_dim,
            num_blocks=critic_num_blocks,
            block_type=critic_block_type,
            rngs=rngs
        )
        self.critic_head = nnx.Linear(
            critic_hidden_dim, 1, rngs=rngs,
            kernel_init=layer_init(1.0),
            bias_init=nnx.initializers.constant(0.0)
        )

        # Actor network
        self.actor_encoder = SimBaEncoder(
            input_dim=obs_dim,
            hidden_dim=actor_hidden_dim,
            num_blocks=actor_num_blocks,
            block_type=actor_block_type,
            rngs=rngs
        )
        self.actor_mean_head = nnx.Linear(
            actor_hidden_dim, action_dim, rngs=rngs,
            kernel_init=layer_init(0.01),
            bias_init=nnx.initializers.constant(0.0)
        )
        self.actor_logstd = nnx.Param(jnp.zeros((1, action_dim)))

    def get_action(self, obs: jnp.ndarray, deterministic: bool = True):
        """Get action from observation (deterministic for evaluation)."""
        x = self.actor_encoder(obs)
        action_mean = self.actor_mean_head(x)
        
        if deterministic:
            return action_mean
        else:
            action_logstd = jnp.broadcast_to(self.actor_logstd.value, action_mean.shape)
            action_std = jnp.exp(action_logstd)
            key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
            probs = tfd.Normal(action_mean, action_std)
            return probs.sample(seed=key)


class PPOPolicy:
    """Policy wrapper for evaluation environment."""
    
    def __init__(self, env, model_path='model_step_4862752.pkl'):
        """
        Initialize PPO policy.
        
        Args:
            env: Evaluation environment
            model_path: Path to trained model checkpoint
        """
        self.action_space = env.action_space
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Initialize observation normalizer
        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))
        
        # Initialize agent
        rngs = nnx.Rngs(0)
        self.agent = Agent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            rngs=rngs,
            critic_hidden_dim=512,
            critic_num_blocks=2,
            critic_block_type="residual",
            actor_hidden_dim=256,
            actor_num_blocks=1,
            actor_block_type="residual"
        )
        
        # Load trained model
        self.load_model(model_path)
        
        print(f"✅ PPO Policy loaded from {model_path}")
        print(f"   Observation dim: {self.obs_dim}")
        print(f"   Action dim: {self.action_dim}")
        print(f"   Normalization: mean={self.obs_rms.mean[:3]}, var={self.obs_rms.var[:3]}")

    def load_model(self, checkpoint_path):
        """Load model weights and normalization statistics."""
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
        
        print(f"Model loaded: step={save_data.get('step', 'unknown')}, "
              f"timestamp={save_data.get('timestamp', 'unknown')}")

    def __call__(self, obs):
        """
        Get action from observation.
        
        Args:
            obs: Raw observation from environment
            
        Returns:
            action: Action to take in environment
        """
        # Normalize observation
        obs_normalized = self.obs_rms.normalize(obs)
        
        # Convert to JAX array and add batch dimension
        obs_jax = jnp.array(obs_normalized).reshape(1, -1)
        
        # Get deterministic action (mean of policy distribution)
        action = self.agent.get_action(obs_jax, deterministic=True)
        
        # Convert back to numpy and remove batch dimension
        action_np = np.array(action).flatten()
        
        # Clip action to valid range
        action_np = np.clip(action_np, self.action_space.low, self.action_space.high)
        
        return action_np


def get_custom_observation(rc, obs_keys):
    """
    Create observation vector from environment observation dict.
    
    Args:
        rc: RemoteConnection instance
        obs_keys: List of observation keys to use
        
    Returns:
        obs: Observation vector
    """
    obs_dict = rc.get_obsdict()
    return rc.obsdict2obsvec(obs_dict, obs_keys)


# Main evaluation loop
time.sleep(10)

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

if LOCAL_EVALUATION:
    rc = RemoteConnection("environment:8085")
else:
    rc = RemoteConnection("localhost:8085")

# Initialize policy
policy = PPOPolicy(rc, model_path='model_step_4862752.pkl')

# Set custom observation keys
shape = get_custom_observation(rc, custom_obs_keys).shape
rc.set_output_keys(custom_obs_keys)

print(f"🎮 Starting evaluation with PPO policy")
print(f"   Observation shape: {shape}")
print(f"   Action space: {rc.action_space}")

flat_completed = None
trial = 0

while not flat_completed:
    flag_trial = None
    ret = 0

    print(f"\n{'='*80}")
    print(f"🏓 Trial {trial}: Resetting environment...")
    
    obs = rc.reset()

    counter = 0
    while not flag_trial:
        # Get action from trained PPO policy
        action = policy(obs)
        
        # Execute action in environment
        base = rc.act_on_environment(action)

        obs = base["feedback"][0]
        reward = base["feedback"][1]
        flag_trial = base["feedback"][2]
        flat_completed = base["eval_completed"]
        ret += reward

        if flag_trial:
            print(f"✅ Trial {trial} completed: Return = {ret:.2f}, Steps = {counter}")
            print("*" * 80)
            break
        
        counter += 1
        
        # Print progress every 100 steps
        if counter % 100 == 0:
            print(f"   Step {counter}: Current return = {ret:.2f}")
    
    trial += 1

print(f"\n🎉 Evaluation completed! Total trials: {trial}")

