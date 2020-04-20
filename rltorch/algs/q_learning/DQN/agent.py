import torch
import torch.nn as nn
import torch.optim as optim

from .model import DQN
from rltorch.algs.q_learning.base.replay import ReplayMemory
from rltorch.algs.q_learning.base.agent import QLearningBaseAgent


class DQNAgent(QLearningBaseAgent):
    """The vanilla DQN Agent (with no target network) """

    def __init__(self, name="DQN", **kwargs):
        super().__init__(name, **kwargs)

        if self.seed:
            torch.manual_seed(self.seed)

        # Neural Network related attributes
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")
        print(f"Using device={self.device}")
        self.network_update_freq = kwargs["network_update_freq"]
        self.dqn = DQN(self.obs_dim,
                       kwargs["hidden_sizes"],
                       self.num_actions).to(self.device)
        print(self.dqn)

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)
        print(self.optimizer)
        self.loss_fn = nn.MSELoss()

        # replay
        self.replay = ReplayMemory(kwargs["replay_size"],
                                   self.obs_dim,
                                   self.device)
        self.updates_done = 0

    def get_action(self, o):
        o = torch.from_numpy(o).float().to(self.device)
        return self.dqn.get_action(o).cpu().item()

    def optimize(self):
        if self.steps_done % self.network_update_freq != 0:
            return None

        if self.steps_done < self.start_steps:
            return 0, 0, 0, 0

        batch = self.replay.sample_batch(self.batch_size)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        # get q_vals for each state and the action performed in that state
        q_vals_raw = self.dqn(s_batch)
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()

        # get target q val = max val of next state
        with torch.no_grad():
            target_q_val_raw = self.dqn(next_s_batch)
            target_q_val, _ = target_q_val_raw.max(1)
            target = r_batch + self.discount*(1-d_batch)*target_q_val

        # calculate loss
        loss = self.loss_fn(q_vals, target)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updates_done += 1

        q_vals_max = q_vals_raw.max(1)[0]
        mean_v = q_vals_max.mean().item()
        max_v = q_vals.max().item()
        mean_td_error = (target - q_vals).abs().mean().item()
        return loss.item(), mean_v, max_v, mean_td_error

    def save_model(self):
        save_path = self.logger.get_save_path(ext=".pth")
        self.dqn.save_DQN(save_path)

    def store(self, o, a, next_o, r, d):
        self.replay.store(o, a, next_o, r, d)
