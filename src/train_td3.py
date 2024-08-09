# reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
import json

import gymnasium as gym
import numpy as np
import torch
import tyro
from stable_baselines3.common.buffers import DictReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from omegaconf import OmegaConf
import itertools

from src.utils.arguments import Args
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

    args = tyro.cli(Args)
    run_name = f"training/seed_{args.seed}/{args.env_id}/{args.env_config_id}/{args.exp_name}"
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
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

    agent = TD3(envs, args, device)

    for key in envs.single_observation_space:
        envs.single_observation_space[key].dtype = envs.observation_space[key].dtype = np.float32
    envs.single_action_space.dtype = envs.action_space.dtype = np.dtype('float32')
    rb = DictReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, infos = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                obs_tensor = {k: torch.Tensor(v).to(device) for k, v in obs.items()}
                actions = agent.actor(**obs_tensor)
                actions += torch.normal(0, agent.actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            writer_info = agent.learn(data, global_step)

            if global_step % 100 == 0:
                for key, value in writer_info.items():
                    writer.add_scalar(key, value, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Save the model
        if (global_step + 1) % (args.total_timesteps // args.num_checkpoints) == 0 or (global_step + 1) == args.total_timesteps:
            model_path = f"runs/{run_name}/checkpoints/{(global_step + 1)}.pth"
            agent.save(model_path)
            with open(model_path.replace('.pth', '_context_rarity_record.json'), 'w') as fout:
                json.dump(agent.context_rarity_record, fout)
            with open(model_path.replace('.pth', '_context_rarity_record_baseline.json'), 'w') as fout:
                json.dump(agent.context_rarity_record_baseline, fout)

    envs.close()
    writer.close()