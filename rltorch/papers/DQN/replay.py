import torch
import numpy as np

from rltorch.papers.DQN.hyperparams import AtariHyperparams as hp


class ReplayMemory:

    def __init__(self, capacity, history_len, width, height, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.h = history_len
        self.s_dims = (self.h, width, height)
        # store as float16 to save space, then we convert when sampling
        self.s_buf = np.zeros((capacity, *self.s_dims),
                              dtype=hp.REPLAY_S_DTYPE)
        self.a_buf = np.zeros((capacity, 1), dtype=np.long)
        next_s_dim = (capacity, 1, width, height)
        self.next_s_buf = np.zeros(next_s_dim,
                                   dtype=hp.REPLAY_S_DTYPE)
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

        # must add last h frames from state to beginning of next state
        s_batch = self.s_buf[sample_idxs]
        next_s_batch = np.concatenate((s_batch[:, 1:],
                                       self.next_s_buf[sample_idxs]),
                                      axis=1)

        batch = [s_batch.astype(np.float32, copy=False),
                 self.a_buf[sample_idxs],
                 next_s_batch.astype(np.float32, copy=False),
                 self.r_buf[sample_idxs],
                 self.done_buf[sample_idxs]]

        # assert s_batch.shape == next_s_batch.shape
        # for i in range(1, self.h):
        #    assert np.array_equal(s_batch[0][i], next_s_batch[0][i-1])

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
