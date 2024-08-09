# reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from omegaconf import OmegaConf
import itertools

from src.utils.arguments import TestArgs
from src.utils.envs import make_env, make_carl_env
from src.models.td3 import TD3


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """
                Ongoing migration: run the following command to install the new dependencies: poetry run pip install "stable_baselines3==2.0.0a1"
            """
        )

    args = tyro.cli(TestArgs)
    for text in args.checkpoint_dir.split("/"):
        if text.startswith("seed_"):
            training_seed = text.split("_")[1]
            break

    run_name = f"test/seed_{training_seed}/{args.env_id}/{args.env_config_id}/{args.exp_name}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    if "CARL" in args.env_id:
        env_config = OmegaConf.load(f"configs/env_configs/{args.env_id}.yaml")
        override_args = env_config.get("override_args", None)
        if override_args is not None:
            vars(args).update(override_args)
        env_config = env_config.get(args.env_config_id, None)
        if env_config is None:
            assert args.env_config_id == "default"
            context_configs = None
        else:
            context_configs = []
            for name, value in env_config.items():
                context_config = {}
                context_config["context_space_type"] = value.pop("context_space_type")
                context_config["context_arguments"] = dict({"name": name}, **value)
                context_configs.append(context_config)

        envs = gym.vector.SyncVectorEnv([make_carl_env(args.env_id, args.seed, 0, args.capture_video, run_name, context_configs, args.n_contexts, args.len_history)])
    else:
        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    checkpoint_idx = args.total_timesteps if args.checkpoint_idx == -1 else args.checkpoint_idx
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoints", f"{checkpoint_idx}.pth")

    agent = TD3(envs, args, device)
    agent.load(checkpoint_path)
    agent.eval()

    envs.single_observation_space.dtype = np.float32

    writer = SummaryWriter(f"runs/{run_name}/checkpoint_{checkpoint_idx}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, infos = envs.reset(seed=args.seed)

    from collections import defaultdict
    test_results = defaultdict(list)
    num_episodes = 0
    global_step = 0

    while num_episodes < args.total_episodes:
        global_step += 1

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            obs_tensor = {k: torch.Tensor(v).to(device) for k, v in obs.items()}
            actions = agent.actor(**obs_tensor)
            actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                episodic_return = info["episode"]["r"]
                episodic_length = info["episode"]["l"]
                test_results["episodic_return"].append(episodic_return)
                test_results["episodic_length"].append(episodic_length)
                print(f"num_episodes={num_episodes}, episodic_return={episodic_return}")
                writer.add_scalar(f"charts/checkpoint_{checkpoint_idx}/episodic_return", episodic_return, num_episodes)
                writer.add_scalar(f"charts/checkpoint_{checkpoint_idx}/episodic_length", episodic_length, num_episodes)

                num_episodes += 1

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step % 100 == 0:
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    print(f"episodic_return: mean={np.mean(test_results['episodic_return'])}, std={np.std(test_results['episodic_return'])}")
    print(f"episodic_length: mean={np.mean(test_results['episodic_length'])}, std={np.std(test_results['episodic_length'])}")
    writer.add_scalar(f"evaluation/seed_{args.seed}/episodic_return_mean", np.mean(test_results["episodic_return"]), checkpoint_idx)
    writer.add_scalar(f"evaluation/seed_{args.seed}/episodic_return_std", np.std(test_results["episodic_return"]), checkpoint_idx)
    writer.add_scalar(f"evaluation/seed_{args.seed}/episodic_length_mean", np.mean(test_results["episodic_length"]), checkpoint_idx)
    writer.add_scalar(f"evaluation/seed_{args.seed}/episodic_length_std", np.std(test_results["episodic_length"]), checkpoint_idx)

    envs.close()
    writer.close()