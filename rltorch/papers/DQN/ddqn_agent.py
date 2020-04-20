import torch

from .dqn_agent import DQNAgent
from rltorch.papers.DQN.hyperparams import AtariHyperparams as hp


class DDQNAgent(DQNAgent):

    def optimize(self):
        if self.steps_done % hp.NETWORK_UPDATE_FREQUENCY != 0:
            return None

        if self.steps_done < hp.REPLAY_START_SIZE:
            return 0, 0, 0, 0

        batch = self.replay.sample_batch(hp.MINIBATCH_SIZE)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        q_vals_raw = self.dqn(s_batch)
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()

        with torch.no_grad():
            max_a_next_s = self.dqn(next_s_batch).max(1)[1].unsqueeze(1)
            target_q_val_raw = self.target_dqn(next_s_batch)
            target_q_vals = target_q_val_raw.gather(1, max_a_next_s).squeeze()
            target = r_batch + (1-d_batch)*hp.DISCOUNT*target_q_vals

        loss = target - q_vals
        loss = loss.clamp(*hp.GRAD_CLIP)
        loss = loss.pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updates_done += 1

        if self.updates_done % hp.TARGET_NETWORK_UPDATE_FREQ == 0:
            self.update_target_net()

        mean_v = q_vals_raw.max(1)[0].mean().item()
        max_v = q_vals.max().item()
        mean_td_error = (target - q_vals).abs().mean().item()
        return loss.item(), mean_v, max_v, mean_td_error
