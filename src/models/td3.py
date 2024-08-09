from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.context_encoder import ContextEncoder


def cosine_distance(a, b):
    a_normalized = F.normalize(a, p=2, dim=-1)
    b_normalized = F.normalize(b, p=2, dim=-1)
    cosine_similarity = torch.matmul(a_normalized, b_normalized.T)
    cosine_distance = 1 - cosine_similarity
    return cosine_distance

def mahalanobis_distance(a, b, cov):
    inv_cov_matrix = torch.linalg.inv(cov)
    diff = a - b
    left = torch.einsum('bi,ij->bj', diff, inv_cov_matrix)
    mahalanobis_distance = torch.sqrt(torch.einsum('bj,bj->b', left, diff))
    return mahalanobis_distance

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, args, context_encoder=None, gt_context_encoder=None):
        super().__init__()
        obs_dim = np.array(env.single_observation_space["obs"].shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        gt_context_dim = np.array(env.single_observation_space["context"].shape).prod()
        context_hidden_dim = args.context_hidden_dim*(args.len_history > 0) + gt_context_dim*(args.use_gt_context)
        input_dim = obs_dim + action_dim + context_hidden_dim

        self.context_encoder = context_encoder
        self.gt_context_encoder = gt_context_encoder

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, action, context=None, history=None):
        x = [obs, action]
        if self.context_encoder is not None:
            x.append(self.context_encoder.get_context(history))
        if self.gt_context_encoder is not None:
            x.append(self.gt_context_encoder(context))

        x = torch.cat(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, args, context_encoder=None, gt_context_encoder=None):
        super().__init__()
        obs_dim = np.array(env.single_observation_space["obs"].shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        gt_context_dim = np.array(env.single_observation_space["context"].shape).prod()
        context_hidden_dim = args.context_hidden_dim*(args.len_history > 0) + gt_context_dim*(args.use_gt_context)
        input_dim = obs_dim + context_hidden_dim

        self.context_encoder = context_encoder
        self.gt_context_encoder = gt_context_encoder

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

    def forward(self, obs, context=None, history=None):
        x = [obs]
        if self.context_encoder is not None:
            x.append(self.context_encoder.get_context(history))
        if self.gt_context_encoder is not None:
            x.append(self.gt_context_encoder(context))

        x = torch.cat(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class TD3(nn.Module):
    def __init__(self, envs, args, device):
        super().__init__()
        self.ema_alpha = 0.05
        self.rarity_alpha = 0.5
        self.context_rarity_record = defaultdict(float)
        self.context_rarity_record_baseline = defaultdict(int)

        self.envs = envs
        self.args = args
        self.device = device

        if args.len_history > 0:
            self.actor_context_encoder = ContextEncoder(envs, args)
            self.qf_context_encoder = ContextEncoder(envs, args)
        else:
            self.actor_context_encoder = None
            self.qf_context_encoder = None
        if args.use_gt_context:
            self.actor_gt_context_encoder = nn.Identity()
            self.qf_gt_context_encoder = nn.Identity()
        else:
            self.actor_gt_context_encoder = None
            self.qf_gt_context_encoder = None

        self.actor = Actor(envs, args, self.actor_context_encoder, self.actor_gt_context_encoder).to(device)
        self.qf1 = QNetwork(envs, args, self.qf_context_encoder, self.qf_gt_context_encoder).to(device)
        self.qf2 = QNetwork(envs, args, self.qf_context_encoder, self.qf_gt_context_encoder).to(device)
        self.qf1_target = QNetwork(envs, args, self.qf_context_encoder, self.qf_gt_context_encoder).to(device)
        self.qf2_target = QNetwork(envs, args, self.qf_context_encoder, self.qf_gt_context_encoder).to(device)
        self.target_actor = Actor(envs, args, self.actor_context_encoder, self.actor_gt_context_encoder).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        if hasattr(args, "learning_rate"):
            self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.learning_rate)
            self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.learning_rate)

        self.qf_average_context_vector = None # torch.zeros((args.context_hidden_dim,)).to(self.device)
        self.actor_average_context_vector = None # torch.zeros((args.context_hidden_dim,)).to(self.device)
        self.actor_average_matmul_context_vector = None

    def learn(self, data, global_step):
        with torch.no_grad():
            clipped_noise = (torch.randn_like(data.actions, device=self.device) * self.args.policy_noise).clamp(
                -self.args.noise_clip, self.args.noise_clip
            ) * self.target_actor.action_scale

            next_state_actions = (self.target_actor(**data.next_observations) + clipped_noise).clamp(
                self.envs.single_action_space.low[0], self.envs.single_action_space.high[0]
            )
            qf1_next_target = self.qf1_target(**data.next_observations, action=next_state_actions)
            qf2_next_target = self.qf2_target(**data.next_observations, action=next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_qf_next_target).view(-1)

        qf1_a_values = self.qf1(**data.observations, action=data.actions).view(-1)
        qf2_a_values = self.qf2(**data.observations, action=data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value, reduction="none")
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value, reduction="none")
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        self.q_optimizer.zero_grad()
        if self.qf_context_encoder is not None:
            qf_context_hidden, _, qf_context_loss = self.qf_context_encoder(**data.observations, action=data.actions, next_obs=data.next_observations["obs"])
            if self.args.bdr:
                ############
                # # EMA update
                # if self.qf_average_context_vector is None:
                #     self.qf_average_context_vector = qf_context_hidden.detach().mean(dim=0, keepdim=True)
                # else:
                #     self.qf_average_context_vector = self.ema_alpha*qf_context_hidden.detach().mean(dim=0, keepdim=True) + (1 - self.ema_alpha)*self.qf_average_context_vector
                # # Rarity
                # qf_rarity = cosine_distance(qf_context_hidden.detach(), self.qf_average_context_vector)
                # qf_rarity = torch.softmax(qf_rarity, dim=0).flatten()*self.args.batch_size # softmax output sum to the batch size
                # qf_loss = ((self.rarity_alpha*qf_rarity*qf_loss) + (1 - self.rarity_alpha)*qf_loss).mean()
                qf_loss = qf_loss.mean()
                ############
            else:
                qf_loss = qf_loss.mean()
            if qf_context_loss is not None:
                qf_context_loss.backward()
        else:
            qf_loss = qf_loss.mean()
        qf_loss.backward()
        self.q_optimizer.step()

        if global_step % self.args.policy_frequency == 0:
            actor_loss = -self.qf1(**data.observations, action=self.actor(**data.observations))
            self.actor_optimizer.zero_grad()
            if self.actor_context_encoder is not None:
                actor_context_hidden, _, actor_context_loss = self.actor_context_encoder(**data.observations, action=data.actions, next_obs=data.next_observations["obs"])
                if self.args.bdr:
                    ############
                    # EMA update
                    batch_wise_context_vector = actor_context_hidden.detach().mean(dim=0, keepdim=True)
                    batch_wise_actor_matmul_context_vector = torch.matmul(actor_context_hidden.detach().T, actor_context_hidden.detach()) / self.args.batch_size
                    if self.actor_average_context_vector is None:
                        self.actor_average_context_vector = batch_wise_context_vector
                        self.actor_average_matmul_context_vector = batch_wise_actor_matmul_context_vector
                    else:
                        self.actor_average_context_vector = self.ema_alpha*batch_wise_context_vector + (1 - self.ema_alpha)*self.actor_average_context_vector
                        self.actor_average_matmul_context_vector = self.ema_alpha*batch_wise_actor_matmul_context_vector + (1 - self.ema_alpha)*self.actor_average_matmul_context_vector
                    # Rarity
                    # actor_rarity = cosine_distance(actor_context_hidden.detach(), self.actor_average_context_vector)
                    cov = self.actor_average_matmul_context_vector - torch.matmul(self.actor_average_context_vector.T, self.actor_average_context_vector)
                    actor_rarity = mahalanobis_distance(actor_context_hidden.detach(), self.actor_average_context_vector, cov)
                    actor_rarity = torch.softmax(actor_rarity, dim=0)*self.args.batch_size # softmax output sum to the batch size
                    actor_loss = ((self.rarity_alpha*actor_rarity*actor_loss) + (1 - self.rarity_alpha)*actor_loss).mean()
                    # Record rarity weights for each context
                    contexts = data.observations['context']
                    for context, rarity_weight in zip(contexts, actor_rarity):
                        self.context_rarity_record[str(tuple(np.round(context.tolist(), 2)))] += rarity_weight.item()
                        self.context_rarity_record_baseline[str(tuple(np.round(context.tolist(), 2)))] += 1
                    ############
                else:
                    actor_loss = actor_loss.mean()
                if actor_context_loss is not None:
                    actor_context_loss.backward()
            else:
                actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the target network
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

        writer_info = {}
        if global_step % 100 == 0:
            writer_info = {
                "losses/qf1_values": qf1_a_values.mean().item(),
                "losses/qf2_values": qf2_a_values.mean().item(),
                "losses/qf1_loss": qf1_loss.mean().item(),
                "losses/qf2_loss": qf2_loss.mean().item(),
                "losses/qf_loss": qf_loss.item() / 2.0,
                "losses/actor_loss": actor_loss.item(),
            }
            if self.qf_context_encoder is not None and qf_context_loss is not None:
                writer_info.update({"losses/qf_context_loss": qf_context_loss.item()})
                writer_info.update({"losses/actor_context_loss": actor_context_loss.item()})

        return writer_info

    def save(self, model_path):
        torch.save({
            "actor": self.actor.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "qf_context_encoder": self.qf_context_encoder.state_dict() if self.qf_context_encoder is not None else None,
            "actor_context_encoder": self.actor_context_encoder.state_dict() if self.actor_context_encoder is not None else None,
            "qf_gt_context_encoder": self.qf_gt_context_encoder.state_dict() if self.qf_gt_context_encoder is not None else None,
            "actor_gt_context_encoder": self.actor_gt_context_encoder.state_dict() if self.actor_gt_context_encoder is not None else None,
        }, model_path)

    def load(self, model_path):
        checkpoint = torch.load(model_path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.qf1.load_state_dict(checkpoint["qf1"])
        self.qf2.load_state_dict(checkpoint["qf2"])
        if self.qf_context_encoder is not None:
            self.qf_context_encoder.load_state_dict(checkpoint["qf_context_encoder"])
            self.actor_context_encoder.load_state_dict(checkpoint["actor_context_encoder"])
        if self.qf_gt_context_encoder is not None:
            self.qf_gt_context_encoder.load_state_dict(checkpoint["qf_gt_context_encoder"])
            self.actor_gt_context_encoder.load_state_dict(checkpoint["actor_gt_context_encoder"])