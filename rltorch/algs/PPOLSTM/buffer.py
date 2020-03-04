import numpy as np
import scipy.signal

import torch


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOBuffer:

    def __init__(self, capacity, obs_dim, gamma=0.99, lam=0.95, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.o_buf = np.zeros((capacity, *obs_dim), dtype=np.float32)
        self.a_buf = np.zeros((capacity, ), dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32)
        self.logp_buf = np.zeros(capacity, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx = 0, 0
        self.ep_ptrs = [0]
        self.max_ep_len = 0

    def store(self, o, a, r, v, logp):
        assert self.ptr < self.capacity
        self.o_buf[self.ptr] = o
        self.a_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.val_buf[self.ptr] = v
        self.logp_buf[self.ptr] = logp
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

        self.max_ep_len = max(self.max_ep_len, self.ptr - self.path_start_idx)
        self.path_start_idx = self.ptr
        self.ep_ptrs.append(self.ptr)

    def get(self):
        """Get all trajectories currently stored"""
        assert self.ptr == self.capacity
        self.ptr, self.path_start_idx = 0, 0

        # normalize advantage
        norm_adv = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)

        # convert batch of observatsions into a list of episodes
        # i.e. one list of observations for each episode
        ep_obs = []
        ep_lengths = []
        for i in range(len(self.ep_ptrs)-1):
            ep_lengths.append(self.ep_ptrs[i+1] - self.ep_ptrs[i])
            o = self.o_buf[self.ep_ptrs[i]:self.ep_ptrs[i+1]]
            ep_obs.append(torch.from_numpy(o).to(self.device))
        # pack the list of episode obs
        # (i.e. list of tensor arrays, where each array contains observations for an episode)
        # into multi-dimension tensor where each episode tensor is packed with 0
        # up to the length of the longest episode
        obs_batch = torch.nn.utils.rnn.pack_sequence(ep_obs, enforce_sorted=False)

        act_batch = torch.from_numpy(self.a_buf).to(self.device)
        ret_batch = torch.from_numpy(self.ret_buf).to(self.device)
        adv_batch = torch.from_numpy(norm_adv).to(self.device)
        logp_batch = torch.from_numpy(self.logp_buf).to(self.device)
        ep_lengths = torch.from_numpy(np.array(ep_lengths, dtype=np.int64))

        data = dict(ep_lens=ep_lengths,
                    obs=obs_batch,
                    act=act_batch,
                    ret=ret_batch,
                    adv=adv_batch,
                    logp=logp_batch)

        self.ep_ptrs = [0]
        self.max_ep_len = 0
        return data
