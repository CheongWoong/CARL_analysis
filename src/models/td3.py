import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.context_encoder import ContextEncoder


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        obs_dim = np.array(env.single_observation_space["obs"].shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        gt_context_dim = np.array(env.single_observation_space["context"].shape).prod()
        context_hidden_dim = args.context_hidden_dim*(args.len_history > 0) + gt_context_dim*(args.use_gt_context)
        input_dim = obs_dim + action_dim + context_hidden_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, action, context_hidden=None, gt_context_hidden=None, **kwargs):
        x = [obs, action]
        if context_hidden is not None:
            x.append(context_hidden)
        if gt_context_hidden is not None:
            x.append(gt_context_hidden)

        x = torch.cat(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        obs_dim = np.array(env.single_observation_space["obs"].shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        gt_context_dim = np.array(env.single_observation_space["context"].shape).prod()
        context_hidden_dim = args.context_hidden_dim*(args.len_history > 0) + gt_context_dim*(args.use_gt_context)
        input_dim = obs_dim + context_hidden_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, obs, context_hidden=None, gt_context_hidden=None, **kwargs):
        x = [obs]
        if context_hidden is not None:
            x.append(context_hidden)
        if gt_context_hidden is not None:
            x.append(gt_context_hidden)

        x = torch.cat(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class TD3(nn.Module):
    def __init__(self, envs, args, device):
        super().__init__()
        self.envs = envs
        self.args = args
        self.device = device

        self.context_encoder_params = []
        if args.len_history > 0:
            self.context_encoder = ContextEncoder(envs, args).to(device)
            self.context_encoder_params += list(self.context_encoder.parameters())
        else:
            self.context_encoder = None
        if args.use_gt_context:
            self.gt_context_encoder = nn.Identity()
            # self.context_encoder_params += list(self.gt_context_encoder.parameters())
        else:
            self.gt_context_encoder = None

        self.actor = Actor(envs, args).to(device)
        self.qf1 = QNetwork(envs, args).to(device)
        self.qf2 = QNetwork(envs, args).to(device)
        self.qf1_target = QNetwork(envs, args).to(device)
        self.qf2_target = QNetwork(envs, args).to(device)
        self.target_actor = Actor(envs, args).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        if hasattr(args, "learning_rate"):
            self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.learning_rate)
            self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.learning_rate)
            if len(self.context_encoder_params) > 0:
                self.context_optimizer = optim.Adam(self.context_encoder_params, lr=args.learning_rate, weight_decay=0.1)
            else:
                self.context_optimizer = None

    def learn(self, data, global_step):
        with torch.no_grad():
            next_context_hidden = None if self.context_encoder is None else self.context_encoder.get_context(data.next_observations["history"])
            next_gt_context_hidden = None if self.gt_context_encoder is None else self.gt_context_encoder(data.next_observations["context"])
            clipped_noise = (torch.randn_like(data.actions, device=self.device) * self.args.policy_noise).clamp(
                -self.args.noise_clip, self.args.noise_clip
            ) * self.target_actor.action_scale

            next_state_actions = (self.target_actor(**data.next_observations, context_hidden=next_context_hidden, gt_context_hidden=next_gt_context_hidden) + clipped_noise).clamp(
                self.envs.single_action_space.low[0], self.envs.single_action_space.high[0]
            )
            qf1_next_target = self.qf1_target(**data.next_observations, action=next_state_actions, context_hidden=next_context_hidden, gt_context_hidden=next_gt_context_hidden)
            qf2_next_target = self.qf2_target(**data.next_observations, action=next_state_actions, context_hidden=next_context_hidden, gt_context_hidden=next_gt_context_hidden)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_qf_next_target).view(-1)

        context_hidden = None if self.context_encoder is None else self.context_encoder.get_context(data.observations["history"])
        gt_context_hidden = None if self.gt_context_encoder is None else self.gt_context_encoder(data.observations["context"])

        qf1_a_values = self.qf1(**data.observations, action=data.actions, context_hidden=context_hidden, gt_context_hidden=gt_context_hidden).view(-1)
        qf2_a_values = self.qf2(**data.observations, action=data.actions, context_hidden=context_hidden, gt_context_hidden=gt_context_hidden).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        if self.context_encoder is not None:
            self.context_optimizer.zero_grad()
            _, _, context_loss = self.context_encoder(**data.observations, action=data.actions, next_obs=data.next_observations["obs"])
            if context_loss is not None:
                context_loss.backward(retain_graph=True)

        # optimize the model
        self.q_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.q_optimizer.step()

        if global_step % self.args.policy_frequency == 0:
            actor_loss = -self.qf1(**data.observations, action=self.actor(**data.observations, context_hidden=context_hidden, gt_context_hidden=gt_context_hidden), context_hidden=context_hidden, gt_context_hidden=gt_context_hidden).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the target network
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

        if self.context_optimizer is not None:
            self.context_optimizer.step()

        writer_info = {}
        if global_step % 100 == 0:
            writer_info = {
                "losses/qf1_values": qf1_a_values.mean().item(),
                "losses/qf2_values": qf2_a_values.mean().item(),
                "losses/qf1_loss": qf1_loss.item(),
                "losses/qf2_loss": qf2_loss.item(),
                "losses/qf_loss": qf_loss.item() / 2.0,
                "losses/actor_loss": actor_loss.item(),
            }
            if self.context_encoder is not None and context_loss is not None:
                writer_info.update({"losses/context_loss": context_loss.item()})

        return writer_info

    def save(self, model_path):
        torch.save({
            "actor": self.actor.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "context_encoder": self.context_encoder.state_dict() if self.context_encoder is not None else None,
            "gt_context_encoder": self.gt_context_encoder.state_dict() if self.gt_context_encoder is not None else None,
        }, model_path)

    def load(self, model_path):
        checkpoint = torch.load(model_path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.qf1.load_state_dict(checkpoint["qf1"])
        self.qf2.load_state_dict(checkpoint["qf2"])
        if self.context_encoder is not None:
            self.context_encoder.load_state_dict(checkpoint["context_encoder"])
        if self.gt_context_encoder is not None:
            self.gt_context_encoder.load_state_dict(checkpoint["gt_context_encoder"])