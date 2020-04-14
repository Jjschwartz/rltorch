import math
import numpy as np


class StatTracker:
    """A class for tracking the running mean and variance.

    Uses the Welford algorithm for running means, var and stdev:
      https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Also tracks the moving mean, var and stdev over a specific window
    size (default=100)
    """

    def __init__(self, window=100):
        self.window = window
        self.ptr = 0
        self.value_buffer = np.zeros(window, dtype=np.float32)
        self.min_val = math.inf
        self.max_val = -math.inf
        self.total = 0
        self.mean = 0
        self.M2 = 0
        self.n = 0

    def update(self, x):
        # handle total running values
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)
        self.total += x
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        # handle moving values
        self.value_buffer[self.ptr] = x
        self.ptr = (self.ptr+1) % self.window

    @property
    def var(self):
        if self.n <= 1:
            return 0
        return self.M2 / self.n

    @property
    def stdev(self):
        return math.sqrt(self.var)

    @property
    def moving_mean(self):
        if self.n < self.window:
            return self.value_buffer[:self.ptr].mean()
        return self.value_buffer.mean()

    @property
    def moving_var(self):
        if self.n < self.window:
            return self.value_buffer[:self.ptr].var()
        return self.value_buffer.var()

    @property
    def moving_stdev(self):
        if self.n < self.window:
            return self.value_buffer[:self.ptr].std()
        return self.value_buffer.std()

    @property
    def moving_max(self):
        if self.n < self.window:
            return self.value_buffer[:self.ptr].max()
        return self.value_buffer.max()

    @property
    def moving_min(self):
        if self.n < self.window:
            return self.value_buffer[:self.ptr].min()
        return self.value_buffer.min()
