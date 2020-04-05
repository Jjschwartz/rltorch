import numpy as np
import scipy.signal

import torch


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOBuffer:

    def __init__(self, capacity, obs_dim, hid_size, gamma=0.99, lam=0.95, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.o_buf = np.zeros((capacity, *obs_dim), dtype=np.float32)
        self.a_buf = np.zeros((capacity, ), dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32)
        self.logp_buf = np.zeros(capacity, dtype=np.float32)
        self.actor_hx_buf = np.zeros((capacity, hid_size), dtype=np.float32)
        self.actor_cx_buf = np.zeros((capacity, hid_size), dtype=np.float32)
        self.critic_hx_buf = np.zeros((capacity, hid_size), dtype=np.float32)
        self.critic_cx_buf = np.zeros((capacity, hid_size), dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx = 0, 0

    def store(self, o, a, r, v, logp, actor_hid, critic_hid):
        assert self.ptr < self.capacity
        self.o_buf[self.ptr] = o
        self.a_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.val_buf[self.ptr] = v
        self.logp_buf[self.ptr] = logp
        self.actor_hx_buf[self.ptr] = actor_hid[0].numpy()
        self.actor_cx_buf[self.ptr] = actor_hid[1].numpy()
        self.critic_hx_buf[self.ptr] = critic_hid[0].numpy()
        self.critic_cx_buf[self.ptr] = critic_hid[1].numpy()
        self.ptr += 1

    def finish_path(self, last_val=0):
        """Call this at end of trajectory """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE - advantage estimate
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # Reward-to-go targets
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        """Get all trajectories currently stored"""
        assert self.ptr == self.capacity
        self.ptr, self.path_start_idx = 0, 0

        # normalize advantage
        norm_adv = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)

        obs_batch = torch.from_numpy(self.o_buf).to(self.device).unsqueeze(0)
        act_batch = torch.from_numpy(self.a_buf).to(self.device)
        ret_batch = torch.from_numpy(self.ret_buf).to(self.device)
        adv_batch = torch.from_numpy(norm_adv).to(self.device)
        logp_batch = torch.from_numpy(self.logp_buf).to(self.device)
        actor_hid_batch = (torch.from_numpy(self.actor_hx_buf).to(self.device).unsqueeze(0),
                           torch.from_numpy(self.actor_cx_buf).to(self.device).unsqueeze(0))
        critic_hid_batch = (torch.from_numpy(self.critic_hx_buf).to(self.device).unsqueeze(0),
                            torch.from_numpy(self.critic_cx_buf).to(self.device).unsqueeze(0))

        data = dict(obs=obs_batch,
                    act=act_batch,
                    ret=ret_batch,
                    adv=adv_batch,
                    logp=logp_batch,
                    actor_hid=actor_hid_batch,
                    critic_hid=critic_hid_batch)
        return data
