"""The PPO Actor and Value NN """
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class Actor(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_sizes,
                 num_actions,
                 activation=nn.ReLU,
                 output_activation=nn.Identity):
        super().__init__()
        layers = [nn.Linear(input_dim[0], hidden_sizes[0]), activation()]
        for l in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[l-1], hidden_sizes[l]))
            layers.append(activation())
        layers.append(nn.Linear(hidden_sizes[-1], num_actions))
        layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x, act=None):
        pi = self.get_pi(x)
        logp_a = None
        if act is not None:
            logp_a = self.get_logp(pi, act)
        return pi, logp_a

    def get_pi(self, obs):
        return Categorical(logits=self.net(obs))

    def get_logp(self, pi, act):
        log_p = pi.log_prob(act)
        return log_p

    def step(self, obs, act):
        pi = self.get_pi(obs)
        logp_a = self.get_logp(pi, act)
        return pi, logp_a

    def act(self, obs):
        with torch.no_grad():
            return self.get_pi(obs).numpy()

    def save_actor(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_actor(self, file_path):
        torch.load_state_dict(torch.load(file_path))
