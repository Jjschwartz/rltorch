import gym
import random
import numpy as np
import torch.optim as optim

from .model import DQN
from .replay import ReplayMemory
import rltorch.papers.DQN.hyperparams as hp
from .preprocess import ImageProcessor, clip_reward


class DQNAgent:

    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.num_actions = self.env.action_space.n
        self.dqn = DQN(self.num_actions)
        self.target_dqn = DQN(self.num_actions)
        self.replay = ReplayMemory(hp.REPLAY_SIZE, hp.STATE_DIMS)
        self.img_processor = ImageProcessor(hp.HEIGHT, hp.WIDTH)
        self.optimizer = optim.RMSprop(self.dqn.parameter(),
                                       lr=hp.LEARNING_RATE,
                                       momentum=hp.GRADIENT_MOMENTUM,
                                       eps=hp.MIN_SQUARED_GRADIENT)

        self.epsilon_schedule = np.linspace(hp.INITIAL_EXPLORATION, hp.FINAL_EXPLORATION,
                                            hp.FINAL_EXPLORATION_FRAME)
        self.steps_done = 0

    def get_action(self, x):
        if self.steps_done < hp.REPLAY_START_SIZE:
            return random.randint(0, self.num_actions-1)

        if self.steps_done < hp.FINAL_EXPLORATION_FRAME:
            epsilon = self.epsilon_schedule[self.steps_done]
        else:
            epsilon = hp.FINAL_EXPLORATION

        if random.random() > epsilon:
            return self.dqn.get_action(x)
        return random.randint(0, self.num_actions-1)

    def optimize(self):
        if self.steps_done < hp.REPLAY_START_SIZE:
            return

        batch = self.replay.sample_batch(hp.MINIBATCH_SIZE)
        s, a, next_s, r, d = batch

    def update_target_net(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def train(self):
        self.steps_done = 0
        s, r, d = self.env.reset(), 0, False
