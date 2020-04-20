import torch

from rltorch.algs.q_learning.DQN.model import DQN
from rltorch.algs.q_learning.DQN.agent import DQNAgent


class DQNTargetAgent(DQNAgent):

    def __init__(self, name="DQNTarget", **kwargs):
        super().__init__(name, **kwargs)

        # Target Neural Network related attributes
        self.target_dqn = DQN(self.obs_dim,
                              kwargs["hidden_sizes"],
                              self.num_actions).to(self.device)
        self.target_update_freq = kwargs["target_update_freq"]

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
            target_q_val_raw = self.target_dqn(next_s_batch)
            target_q_val, _ = target_q_val_raw.max(1)
            target = r_batch + self.discount*(1-d_batch)*target_q_val

        # calculate loss
        loss = self.loss_fn(q_vals, target)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updates_done += 1

        if self.updates_done % self.target_update_freq == 0:
            self.update_target_net()

        q_vals_max = q_vals_raw.max(1)[0]
        mean_v = q_vals_max.mean().item()
        max_v = q_vals.max().item()
        mean_td_error = (target - q_vals).abs().mean().item()
        return loss.item(), mean_v, max_v, mean_td_error

    def update_target_net(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())
