from .dqn_agent import DQNAgent
from rltorch.papers.DQN.hyperparams import AtariHyperparams as hp

RENDER = False


class DDQNAgent(DQNAgent):

    def optimize(self):
        if self.steps_done < hp.REPLAY_START_SIZE:
            return 0, 0, 0, 0

        batch = self.replay.sample_batch(hp.MINIBATCH_SIZE)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        # print("s_batch.shape", s_batch.shape)

        q_vals_raw = self.dqn(s_batch)
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()
        # print("q_vals.shape", q_vals.shape)

        max_a_next_s = self.dqn(next_s_batch).max(1)[1].unsqueeze(1)
        # print("max_a_next_s.shape", max_a_next_s.shape)
        target_q_val_raw = self.target_dqn(next_s_batch)
        # print("target_q_val_raw.shape", target_q_val_raw.shape)
        target_q_vals = target_q_val_raw.gather(1, max_a_next_s).squeeze()
        # print("target_q_vals.shape", target_q_vals.shape)
        target = r_batch + hp.DISCOUNT*(1-d_batch)*target_q_vals
        # print("target.shape", target.shape)

        loss = self.loss_fn(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn.parameters():
            # clip squared gradient
            param.grad.data.clamp_(*hp.GRAD_CLIP)
        self.optimizer.step()

        loss_value = loss.item()
        q_vals_max = q_vals_raw.max(1)[0]
        mean_v = q_vals_max.mean().item()
        max_v = q_vals.max().item()
        mean_td_error = (target - q_vals).abs().mean().item()
        return loss_value, mean_v, max_v, mean_td_error