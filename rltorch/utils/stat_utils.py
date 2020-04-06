import math


class StatTracker:
    """A class for tracking the running mean and variance.

    Uses the Welford algorithm for running means, var and stdev:
      https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    def __init__(self):
        self.min_val = math.inf
        self.max_val = -math.inf
        self.total = 0
        self.mean = 0
        self.M2 = 0
        self.n = 0

    def update(self, x):
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)
        self.total += x
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self):
        return self.M2 / self.n

    @property
    def stdev(self):
        return math.sqrt(self.var)
