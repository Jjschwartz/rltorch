import torch
import numpy as np


class ReplayMemory:

    def __init__(self, capacity, history_len, width, height, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.h = history_len
        self.s_dims = (self.h, width, height)
        # store as float16 to save space, then we convert when sampling
        self.s_buf = np.zeros((capacity, *self.s_dims), dtype=np.uint8)
        self.a_buf = np.zeros((capacity, 1), dtype=np.long)
        next_s_dim = (capacity, 1, width, height)
        self.next_s_buf = np.zeros(next_s_dim, dtype=np.uint8)
        self.r_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, s, a, next_s, r, done):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample_batch(self, batch_size):
        sample_idxs = np.random.choice(self.size, batch_size)

        # convert from uint to float
        s_batch = self.s_buf[sample_idxs].astype(np.float32, copy=False)
        ns_sample = self.next_s_buf[sample_idxs].astype(np.float32, copy=False)
        # must add last h frames from state to beginning of next state
        next_s_batch = np.concatenate((s_batch[:, 1:], ns_sample), axis=1)

        batch = [s_batch,
                 self.a_buf[sample_idxs],
                 next_s_batch,
                 self.r_buf[sample_idxs],
                 self.done_buf[sample_idxs]]

        return [torch.from_numpy(buf).to(self.device) for buf in batch]

    def display_memory_usage(self):
        total = self.s_buf.nbytes
        total += self.a_buf.nbytes
        total += self.next_s_buf.nbytes
        total += self.r_buf.nbytes
        total += self.done_buf.nbytes
        print("Replay Memory Usage:")
        print(f"\tState buffer = {self.s_buf.nbytes / 1e6} MBytes")
        print(f"\tAction buffer = {self.a_buf.nbytes / 1e6} MBytes")
        print(f"\tNext State buffer = {self.next_s_buf.nbytes / 1e6} MBytes")
        print(f"\tReward buffer = {self.r_buf.nbytes / 1e6} MBytes")
        print(f"\tDone buffer = {self.done_buf.nbytes / 1e6} MBytes")
        print(f"\tTotal = {total / 1e6} MBytes")
