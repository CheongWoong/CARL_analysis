from dataclasses import dataclass


@dataclass
class Args:
    exp_name: str = ""
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CARLPendulum"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    # Additional arguments
    num_checkpoints: int = 10
    """the number of checkpoints to save the model"""
    device_id: int = 0
    """the gpu id"""
    env_config_id: str = "train"
    """the id for context config (default, train, test)"""
    n_contexts: int = 0
    """the number of contexts to be sampled"""
    len_history: int = 0
    """the length of history for context encoder input"""
    context_objective: str = "none"
    """the training objective for context encoder (e.g. none, osi, dm, ...)"""
    use_gt_context: bool = False
    """whether to use the groundtruth context or not"""
    context_hidden_dim: int = 10
    """the dimension for context hidden vector"""
    bdr: bool = False
    """whether to use BDR (balanced domain randomization) algorithm"""


@dataclass
class TestArgs:
    exp_name: str = ""
    """the name of this experiment"""
    seed: int = 101
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CARLPendulum"
    """the id of the environment"""
    total_episodes: int = 100
    """the number of episodes for evaluation"""

    # Additional arguments
    checkpoint_idx: int = -1
    checkpoint_dir: str = ""
    """the directory for the checkpoint"""
    device_id: int = 0
    """the gpu id"""
    env_config_id: str = "train"
    """the id for context config (default, train, test)"""
    n_contexts: int = 0
    """the number of contexts to be sampled"""
    len_history: int = 0
    """the length of history for context encoder input"""
    context_objective: str = "none"
    """the training objective for context encoder (e.g. none, osi, dm, ...)"""
    use_gt_context: bool = False
    """whether to use the groundtruth context or not"""
    context_hidden_dim: int = 10
    """the dimension for context hidden vector"""