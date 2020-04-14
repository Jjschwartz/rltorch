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
from rltorch.utils.stat_utils import StatTracker

PAUSE_DISPLAY = False
DISPLAY_DELAY = 0.01


class DQNAgent:
    """The vanilla DQN Agent (with no target network) """

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
        if "exp_name" in kwargs and kwargs["exp_name"]:
            logger_name = kwargs["exp_name"]
        self.logger = RLLogger(self.env_name, logger_name)
        self.setup_logger()
        self.logger.save_config(kwargs)
        self.return_tracker = StatTracker()

        # Neural Network related attributes
        self.dqn = DQN(self.obs_dim,
                       kwargs["hidden_sizes"],
                       self.num_actions).to(self.device)
        print(self.dqn)

        # self.optimizer = optim.Adam(self.dqn.parameters(), lr=kwargs["lr"])
        self.optimizer = optim.RMSprop(self.dqn.parameters(),
                                       lr=0.00025,
                                       momentum=0.0,
                                       alpha=0.95,
                                       eps=0.01)
        print(self.optimizer)
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
        self.network_update_freq = kwargs["network_update_freq"]
        self.model_save_freq = kwargs["model_save_freq"]
        self.steps_done = 0

    def setup_logger(self):
        self.logger.add_header("episode")
        self.logger.add_header("seed")
        self.logger.add_header("steps_done")
        self.logger.add_header("epsilon")
        self.logger.add_header("episode_return")
        self.logger.add_header("episode_loss")
        self.logger.add_header("episode_mean_v")
        self.logger.add_header("episode_mean_td_error")
        self.logger.add_header("mean_episode_return")
        self.logger.add_header("min_episode_return")
        self.logger.add_header("max_episode_return")
        self.logger.add_header("episode_return_stdev")
        self.logger.add_header("time")

    def get_epsilon(self):
        if self.steps_done < self.start_steps:
            return 1.0
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        return self.final_epsilon

    def get_action(self, x):
        if random.random() > self.get_epsilon():
            x = torch.from_numpy(x).float().to(self.device)
            return self.dqn.get_action(x).cpu().item()
        return random.randint(0, self.num_actions-1)

    def optimize(self):
        if self.steps_done < self.start_steps:
            return 0, 0, 0

        batch = self.replay.sample_batch(self.batch_size)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        # get q_vals for each state and the action performed in that state
        q_vals_raw = self.dqn(s_batch)
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()

        # get target q val = max val of next state
        with torch.no_grad():
            target_q_val_raw = self.dqn(next_s_batch)
            target_q_val, _ = target_q_val_raw.max(1)
            target = r_batch + self.discount*(1-d_batch)*target_q_val

        # calculate loss
        loss = self.loss_fn(q_vals, target)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_value = loss.item()
        mean_v = target_q_val.mean().item()
        mean_td_error = (target - q_vals).abs().mean().item()
        return (loss_value, mean_v, mean_td_error)

    def train(self):
        print("Starting training")
        self.steps_done = 0

        num_episodes = 0

        display_freq = min(100, int(self.training_steps // 10))

        while self.steps_done < self.training_steps:
            start_time = time.time()
            self.run_episode()
            num_episodes += 1

            self.logger.log("episode", num_episodes)
            self.logger.log("seed", self.seed)
            self.logger.log("steps_done", self.steps_done)
            self.logger.log("epsilon", self.get_epsilon())
            self.logger.log("time", time.time()-start_time)

            display = num_episodes % display_freq == 0
            self.logger.flush(display)
            if display:
                self.display_run()

        self.logger.flush(True)
        self.display_run()
        print("Training complete")

    def run_episode(self):
        o = self.env.reset()
        done = False

        episode_return = 0
        episode_loss = 0

        while not done and self.steps_done < self.training_steps:
            a = self.get_action(o)
            next_o, r, done, _ = self.env.step(a)
            assert o is not next_o

            self.replay.store(o, a, next_o, r, done)
            o = next_o
            episode_return += r
            self.steps_done += 1

            if self.steps_done % self.network_update_freq == 0:
                episode_loss, mean_v, mean_td_error = self.optimize()

            if self.model_save_freq is not None and \
               self.steps_done % self.model_save_freq == 0:
                save_path = self.logger.get_save_path(ext=".pth")
                self.dqn.save_DQN(save_path)

        self.return_tracker.update(episode_return)
        self.logger.log("episode_return", episode_return)
        self.logger.log("episode_loss", episode_loss)
        self.logger.log("episode_mean_v", mean_v)
        self.logger.log("episode_mean_td_error", mean_td_error)
        self.logger.log("mean_episode_return", self.return_tracker.mean)
        self.logger.log("min_episode_return", self.return_tracker.min_val)
        self.logger.log("max_episode_return", self.return_tracker.max_val)
        self.logger.log("episode_return_stdev", self.return_tracker.stdev)

    def display_run(self, step_limit=1000):
        print("Running policy")
        o = self.env.reset()
        done = False
        episode_return = 0
        self.env.render()
        t = 0
        while not done and t < step_limit:
            a = self.get_action(o)
            o, r, done, _ = self.env.step(a)
            episode_return += r
            t += 1
            self.env.render()
            if PAUSE_DISPLAY:
                input(f"Step = o: {o}, a: {a}, r: {r:.4f}, d: {done}")
            else:
                time.sleep(DISPLAY_DELAY)
            if t > 0 and t % int(step_limit // 10) == 0:
                print(f"Steps taken = {t}")
        print("Episode done:")
        print(f"Reward = {episode_return}")
