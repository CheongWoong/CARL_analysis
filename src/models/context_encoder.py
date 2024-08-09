import numpy as np
import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    def __init__(self, env, args):
        super().__init__()

        self.context_objective = args.context_objective

        obs_dim = np.array(env.single_observation_space["obs"].shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        gt_context_dim = np.array(env.single_observation_space["context"].shape).prod()

        self.context_encoder = nn.Sequential(
            nn.Linear((obs_dim + action_dim)*args.len_history, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, args.context_hidden_dim),
        )
        
        if self.context_objective == "osi":
            self.task_head = nn.Sequential(
                nn.Linear(args.context_hidden_dim, args.context_hidden_dim),
                nn.Tanh(),
                nn.Linear(args.context_hidden_dim, gt_context_dim),
            )
        elif self.context_objective == "dm":
            self.task_head = nn.Sequential(
                nn.Linear(args.context_hidden_dim + obs_dim + action_dim, args.context_hidden_dim +  + obs_dim + action_dim),
                nn.Tanh(),
                nn.Linear(args.context_hidden_dim  + obs_dim + action_dim, obs_dim),
            )
        elif self.context_objective == 'none':
            self.task_head = None

        self.mse_loss = nn.MSELoss()

    def get_context(self, history):
        return self.context_encoder(history)

    def forward(self, obs, action, next_obs, context=None, history=None):
        context_hidden = self.get_context(history)

        # context regularization
        lambda_reg = 0.01
        embedding_norm = torch.norm(context_hidden, p=2)
        regularization_loss = lambda_reg * embedding_norm ** 2

        if self.context_objective == "osi":
            prediction = self.task_head(context_hidden)
            loss = torch.sqrt(self.mse_loss(prediction, context))
        elif self.context_objective == "dm":
            prediction = self.task_head(torch.cat([context_hidden, obs, action], dim=1))
            loss = torch.sqrt(self.mse_loss(prediction, next_obs))
        else:
            prediction, loss = None, None
        total_loss = loss

        return context_hidden, prediction, total_loss