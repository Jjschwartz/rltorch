"""The PPO Actor and Value NN """
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class PPOLSTMActor(nn.Module):

    def __init__(self, input_dim, hidden_size, num_actions, output_activation):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_dim, hidden_size)
        self.out = nn.Linear(hidden_size, num_actions)
        self.output_activation = output_activation()

    def get_init_hidden(self, batch_size=1):
        return (torch.randn(1, batch_size, self.hidden_size),
                torch.randn(1, batch_size, self.hidden_size))

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.output_activation(self.out(x))
        return x, hidden

    def get_pi(self, x, hidden):
        x, hidden = self.forward(x, hidden)
        return Categorical(logits=x), hidden

    def get_logp(self, pi, act):
        log_p = pi.log_prob(act)
        return log_p

    def step(self, obs, act, hidden):
        pi, hidden = self.get_pi(obs, hidden)
        logp_a = self.get_logp(pi, act)
        return pi, logp_a, hidden


class PPOLSTMCritic(nn.Module):

    def __init__(self, input_dim, hidden_size, output_activation):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_dim, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.output_activation = output_activation()

    def get_init_hidden(self, batch_size=1):
        return (torch.randn(1, batch_size, self.hidden_size),
                torch.randn(1, batch_size, self.hidden_size))

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.output_activation(self.out(x))
        # removes last dimension
        return torch.squeeze(x, -1), hidden


class PPOLSTMActorCritic(nn.Module):

    def __init__(self, obs_dim, hidden_size, num_actions, output_activation=nn.Identity):
        super().__init__()
        self.actor = PPOLSTMActor(obs_dim, hidden_size, num_actions, output_activation)
        self.critic = PPOLSTMCritic(obs_dim, hidden_size, output_activation)

    def get_init_hidden(self):
        return self.actor.get_init_hidden(), self.critic.get_init_hidden()

    def step(self, obs, hidden_actor, hidden_critic):
        with torch.no_grad():
            pi, new_hidden_actor = self.actor.get_pi(obs, hidden_actor)
            a = pi.sample()
            logp_a = self.actor.get_logp(pi, a)
            v, new_hidden_critic = self.critic(obs, hidden_critic)
        return a.numpy(), v.numpy(), logp_a.numpy(), new_hidden_actor, new_hidden_critic

    def act(self, obs, hidden):
        with torch.no_grad():
            obs = obs.view(1, 1, -1)
            pi, new_hidden = self.actor.get_pi(obs, hidden)
            a = pi.sample()
            return a.numpy(), (new_hidden[0].numpy(), new_hidden[1].numpy())

    def get_value(self, obs, hidden):
        with torch.no_grad():
            v, new_hidden = self.critic(obs, hidden)
            return v.numpy(), (new_hidden[0].numpy(), new_hidden[1].numpy())

    def save_AC(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_AC(self, file_path):
        torch.load_state_dict(torch.load(file_path))
