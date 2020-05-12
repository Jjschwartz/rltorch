import numpy as np
import scipy.signal

import torch


def discount_cumsum(x, discount):
    return scipy.signal.lfilter(
        [1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:

    def __init__(self, capacity, obs_dim, gamma=0.99, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.o_buf = np.zeros((capacity, *obs_dim), dtype=np.float32)
        self.a_buf = np.zeros((capacity, ), dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.logp_buf = np.zeros(capacity, dtype=np.float32)
        self.gamma = gamma
        self.ptr, self.path_start_idx = 0, 0

    def store(self, o, a, r, logp):
        assert self.ptr < self.capacity
        self.o_buf[self.ptr] = o
        self.a_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """Call this at end of trajectory """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)

        # Reward-to-go targets
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        """Get all trajectories currently stored"""
        assert self.ptr == self.capacity
        self.ptr, self.path_start_idx = 0, 0

        data = [self.o_buf,
                self.a_buf,
                self.ret_buf,
                self.logp_buf]
        return [torch.from_numpy(v).to(self.device) for v in data]
