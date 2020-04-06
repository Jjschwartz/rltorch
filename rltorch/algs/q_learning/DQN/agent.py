import gym
import time
import random
import numpy as np
from pprint import pprint


import torch
import torch.nn as nn
import torch.optim as optim

from .model import DQN
from .replay import ReplayMemory
from rltorch.utils.rl_logger import RLLogger


class DQNAgent:

    def __init__(self, **kwargs):
        print("\nDQN with config:")
        pprint(kwargs)

        self.seed = kwargs["seed"]
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.env_name = kwargs["env_name"]
        self.env = gym.make(self.env_name)
        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape

        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")
        print(f"Using device={self.device}")

        self.replay = ReplayMemory(kwargs["replay_size"],
                                   self.obs_dim,
                                   self.device)
        logger_name = "dqn"
        if "exp_name" in kwargs:
            logger_name = kwargs["exp_name"]
        self.logger = RLLogger(self.env_name, logger_name)
        self.setup_logger()
        self.logger.save_config(kwargs)

        # Neural Network related attributes
        self.dqn = DQN(self.obs_dim,
                       kwargs["hidden_sizes"],
                       self.num_actions).to(self.device)
        self.target_dqn = DQN(self.obs_dim,
                              kwargs["hidden_sizes"],
                              self.num_actions).to(self.device)
        print(self.dqn)

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=kwargs["lr"])
        self.loss_fn = nn.MSELoss()

        # Training related attributes
        self.exploration_steps = kwargs["exploration"]
        self.final_epsilon = kwargs["final_epsilon"]
        self.epsilon_schedule = np.linspace(kwargs["init_epsilon"],
                                            self.final_epsilon,
                                            self.exploration_steps)
        self.start_steps = kwargs["start_steps"]
        self.batch_size = kwargs["batch_size"]
        self.discount = kwargs["gamma"]
        self.training_steps = kwargs["training_steps"]
        self.target_update_freq = kwargs["target_update_freq"]
        self.network_update_freq = kwargs["network_update_freq"]
        self.model_save_freq = kwargs["model_save_freq"]
        self.steps_done = 0

    def setup_logger(self):
        self.logger.add_header("episode")
        self.logger.add_header("seed")
        self.logger.add_header("steps_done")
        self.logger.add_header("episode_return")
        self.logger.add_header("episode_loss")
        self.logger.add_header("time")

    def get_action(self, x):
        if self.steps_done < self.start_steps:
            return random.randint(0, self.num_actions-1)

        if self.steps_done < self.exploration_steps:
            epsilon = self.epsilon_schedule[self.steps_done]
        else:
            epsilon = self.final_epsilon

        if random.random() > epsilon:
            x = torch.from_numpy(x).float().to(self.device)
            return self.dqn.get_action(x)
        return random.randint(0, self.num_actions-1)

    def optimize(self):
        if self.steps_done < self.start_steps:
            return 0

        batch = self.replay.sample_batch(self.batch_size)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        # get q_vals for each state and the action performed in that state
        q_vals_raw = self.dqn(s_batch)
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()

        # get target q val = max val of next state
        with torch.no_grad():
            target_q_val_raw = self.target_dqn(next_s_batch)
            target_q_val, _ = target_q_val_raw.max(1)
            target = r_batch + self.discount*(1-d_batch)*target_q_val

        # calculate loss
        loss = self.loss_fn(q_vals, target)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn.parameters():
            # clip squared gradient
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        loss_value = loss.item()
        return loss_value

    def update_target_net(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def train(self):
        print("Starting training")
        self.steps_done = 0

        num_episodes = 0
        episode_returns = []

        display_freq = min(100, int(self.training_steps // 10))

        while self.steps_done < self.training_steps:
            start_time = time.time()
            ep_return, ep_loss = self.run_episode()
            episode_returns.append(ep_return)
            num_episodes += 1

            self.logger.log("episode", num_episodes)
            self.logger.log("seed", self.seed)
            self.logger.log("steps_done", self.steps_done)
            self.logger.log("episode_return", ep_return)
            self.logger.log("episode_loss", ep_loss)
            self.logger.log("time", time.time()-start_time)

            display = num_episodes % display_freq == 0
            self.logger.flush(display)

        print("Training complete")

    def run_episode(self):
        o = self.env.reset()
        done = False

        episode_return = 0
        episode_loss = 0

        while not done and self.steps_done < self.training_steps:
            a = self.get_action(o)
            next_o, r, done, _ = self.env.step(a)

            self.replay.store(o, a, next_o, r, done)
            o = next_o
            episode_return += r
            self.steps_done += 1

            if self.steps_done % self.network_update_freq == 0:
                episode_loss = self.optimize()

            if self.steps_done % self.target_update_freq == 0:
                self.update_target_net()

            if self.model_save_freq is not None and \
               self.steps_done % self.model_save_freq == 0:
                save_path = self.logger.get_save_path(ext=".pth")
                self.dqn.save_DQN(save_path)

        return episode_return, episode_loss