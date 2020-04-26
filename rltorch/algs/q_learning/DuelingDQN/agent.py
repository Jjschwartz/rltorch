import torch.optim as optim

from rltorch.algs.q_learning.DDQN.agent import DDQNAgent
from rltorch.algs.q_learning.DuelingDQN.model import DuelingDQN


class DuelingDQNAgent(DDQNAgent):

    def __init__(self, name="DuelingDQN", **kwargs):
        super().__init__(name, **kwargs)

        # just need to overwrite the NN models
        self.dqn = DuelingDQN(self.obs_dim,
                              kwargs["hidden_sizes"],
                              self.num_actions,
                              kwargs["dueling_sizes"]).to(self.device)
        self.target_dqn = DuelingDQN(self.obs_dim,
                                     kwargs["hidden_sizes"],
                                     self.num_actions,
                                     kwargs["dueling_sizes"]).to(self.device)
        print(self.dqn)
        # Must also reinitialize optimized with correct params
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)
