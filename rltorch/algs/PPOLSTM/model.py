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
        self.reset()

    def reset(self, batch_size=1):
        # 1st num = num_layers*num_directions = 1*1 = 1
        # 2nd num = batch size
        # 3rd num = hidden size
        self.hidden = (torch.randn(1, batch_size, self.hidden_size),
                       torch.randn(1, batch_size, self.hidden_size))

    def forward(self, x):
        hx, cx = self.hidden
        x, (hx, cx) = self.lstm(x, (hx, cx))
        x = self.output_activation(self.out(x))
        self.hidden = (hx, cx)
        return x

    def batch_forward(self, batch_x, ep_lens, flat_size):
        """Perform a batched forward step.

        batch_x : a PackedSequence containing observations for each episode
        """
        self.reset(len(ep_lens))
        hx, cx = self.hidden
        x, (hx, cx) = self.lstm(batch_x, (hx, cx))

        # flatten lstm output
        unpacked_x, ep_lens = torch.nn.utils.rnn.pad_packed_sequence(x)
        flatten_x = torch.zeros((flat_size, self.hidden_size), dtype=torch.float)
        new_x = unpacked_x.transpose(0, 1)
        ptr = 0
        for e, l in enumerate(ep_lens):
            flatten_x[ptr:ptr+l] = new_x[e][:l]
            ptr += l

        return self.output_activation(self.out(flatten_x))

    def get_pi(self, x):
        x = self.forward(x)
        return Categorical(logits=x)

    def get_logp(self, pi, act):
        log_p = pi.log_prob(act)
        return log_p

    def step(self, obs, act):
        pi = self.get_pi(obs)
        logp_a = self.get_logp(pi, act)
        return pi, logp_a

    def step_batch(self, obs_batch, act_batch, ep_lens):
        x = self.batch_forward(obs_batch, ep_lens, len(act_batch))
        pi = Categorical(logits=x)
        log_p = self.get_logp(pi, act_batch)
        return pi, log_p


class PPOLSTMCritic(nn.Module):

    def __init__(self, input_dim, hidden_size, output_activation):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_dim, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.output_activation = output_activation()
        self.reset()

    def reset(self, batch_size=1):
        self.hidden = (torch.randn(1, batch_size, self.hidden_size),
                       torch.randn(1, batch_size, self.hidden_size))

    def forward(self, x):
        hx, cx = self.hidden
        x, (hx, cx) = self.lstm(x, (hx, cx))
        x = self.output_activation(self.out(x))
        self.hidden = (hx, cx)
        # removes last dimension
        return torch.squeeze(x, -1)

    def batch_forward(self, batch_x, ep_lens, flat_size):
        self.reset(len(ep_lens))
        hx, cx = self.hidden
        x, (hx, cx) = self.lstm(batch_x, (hx, cx))

        # flatten lstm output
        unpacked_x, ep_lens = torch.nn.utils.rnn.pad_packed_sequence(x)
        flatten_x = torch.zeros((flat_size, self.hidden_size), dtype=torch.float)
        new_x = unpacked_x.transpose(0, 1)
        ptr = 0
        for e, l in enumerate(ep_lens):
            flatten_x[ptr:ptr+l] = new_x[e][:l]
            ptr += l

        return self.output_activation(self.out(flatten_x))

    def step_batch(self, obs_batch, flat_size, ep_lens):
        x = self.batch_forward(obs_batch, ep_lens, flat_size)
        return torch.squeeze(x, -1)


class PPOLSTMActorCritic(nn.Module):

    def __init__(self, obs_dim, hidden_size, num_actions, output_activation=nn.Identity):
        super().__init__()
        self.actor = PPOLSTMActor(obs_dim, hidden_size, num_actions, output_activation)
        self.critic = PPOLSTMCritic(obs_dim, hidden_size, output_activation)

    def reset(self, batch_size=1):
        self.actor.reset(batch_size)
        self.critic.reset(batch_size)

    def step(self, obs):
        with torch.no_grad():
            pi = self.actor.get_pi(obs)
            a = pi.sample()
            logp_a = self.actor.get_logp(pi, a)
            v = self.critic(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        with torch.no_grad():
            obs = obs.view(1, 1, -1)
            pi = self.actor.get_pi(obs)
            a = pi.sample()
            return a.numpy()

    def get_value(self, obs):
        with torch.no_grad():
            v = self.critic(obs)
            return v.numpy()

    def save_AC(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_AC(self, file_path):
        torch.load_state_dict(torch.load(file_path))
