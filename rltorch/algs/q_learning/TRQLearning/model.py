import numpy as np
from pprint import pprint

import rltorch.utils.file_utils as futils


class TabularQ:
    """Tabular Q-Function """

    def __init__(self, input_dim, num_actions):
        self.q_func = dict()
        self.num_actions = num_actions

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = str(x.astype(np.int))
        if x not in self.q_func:
            self.q_func[x] = np.zeros(self.num_actions,
                                      dtype=np.float32)
        return self.q_func[x]

    def forward_batch(self, x_batch):
        return np.asarray([self.forward(x) for x in x_batch])

    def update(self, s_batch, a_batch, delta_batch):
        for s, a, delta in zip(s_batch, a_batch, delta_batch):
            q_vals = self.forward(s)
            q_vals[a] += delta

    def save(self, file_path):
        futils.write_yaml(file_path, self.q_func)

    def load(self, file_path):
        self.q_func = futils.load_yaml(file_path)

    def get_action(self, x):
        return int(self.forward(x).argmax())

    def display(self):
        pprint(self.q_func)
