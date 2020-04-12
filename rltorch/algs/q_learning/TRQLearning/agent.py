import gym
import time
import random
import gym_maze
import numpy as np
from pprint import pprint

from .model import TabularQ
from .replay import ReplayMemory

from rltorch.utils.rl_logger import RLLogger
from rltorch.utils.stat_utils import StatTracker


class TabularReplayAgent:
    """A Tabular Q-learning Agent that uses replay """

    def __init__(self, **kwargs):
        print("\nTabularReplayAgent with config:")
        pprint(kwargs)

        self.seed = kwargs["seed"]
        if self.seed is not None:
            np.random.seed(self.seed)

        self.env_name = kwargs["env_name"]
        self.env = gym.make(self.env_name)
        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape

        self.replay = ReplayMemory(kwargs["replay_size"],
                                   self.obs_dim)
        logger_name = "TRQ-Learning"
        if "exp_name" in kwargs and kwargs["exp_name"]:
            logger_name = kwargs["exp_name"]
        self.logger = RLLogger(self.env_name, logger_name)
        self.setup_logger()
        self.logger.save_config(kwargs)
        self.return_tracker = StatTracker()

        # Our tabular Q-value function
        self.tqf = TabularQ(self.obs_dim, self.num_actions)

        # Training related attributes
        self.lr = kwargs["lr"]
        self.exploration_steps = kwargs["exploration"]
        self.final_epsilon = kwargs["final_epsilon"]
        self.epsilon_schedule = np.linspace(kwargs["init_epsilon"],
                                            self.final_epsilon,
                                            self.exploration_steps)
        self.start_steps = kwargs["start_steps"]
        self.batch_size = kwargs["batch_size"]
        self.discount = kwargs["gamma"]
        self.training_steps = kwargs["training_steps"]
        self.function_update_freq = kwargs["function_update_freq"]
        self.model_save_freq = kwargs["model_save_freq"]
        self.steps_done = 0

    def setup_logger(self):
        self.logger.add_header("episode")
        self.logger.add_header("seed")
        self.logger.add_header("steps_done")
        self.logger.add_header("episode_return")
        self.logger.add_header("episode_mean_v")
        self.logger.add_header("episode_mean_td_error")
        self.logger.add_header("mean_episode_return")
        self.logger.add_header("min_episode_return")
        self.logger.add_header("max_episode_return")
        self.logger.add_header("episode_return_stdev")
        self.logger.add_header("time")

    def get_action(self, x):
        if self.steps_done < self.start_steps:
            return random.randint(0, self.num_actions-1)

        if self.steps_done < self.exploration_steps:
            epsilon = self.epsilon_schedule[self.steps_done]
        else:
            epsilon = self.final_epsilon

        if random.random() > epsilon:
            return self.tqf.get_action(x)
        return random.randint(0, self.num_actions-1)

    def optimize(self):
        if self.steps_done < self.start_steps:
            return 0, 0

        batch = self.replay.sample_batch(self.batch_size)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        # get q_vals for each state and the action performed in that state
        q_vals_raw = self.tqf.forward_batch(s_batch)
        q_vals = np.take_along_axis(q_vals_raw, a_batch, axis=1).squeeze()
        # print("q_vals_raw.shape:", q_vals_raw.shape)
        # print("q_vals.shape:", q_vals.shape)

        # get target q val = max val of next state
        target_q_val_raw = self.tqf.forward_batch(next_s_batch)
        target_q_val = target_q_val_raw.max(axis=1)
        target = r_batch+self.discount*(1-d_batch)*target_q_val
        # target = r_batch+self.discount*target_q_val

        # print("target_q_val.shape:", target_q_val.shape)
        # print("target.shape:", target.shape)

        # calculate td_error
        td_error = target-q_vals
        # print("td_error.shape:", td_error.shape)

        # the update for each Q-A state
        td_delta = self.lr*td_error
        # print("td_delta", td_delta.shape)

        # perform update to function
        self.tqf.update(s_batch, a_batch, td_delta)

        mean_v = target_q_val.mean().item()
        mean_td_error = np.absolute(td_error).mean().item()
        return (mean_v, mean_td_error)

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
            self.logger.log("time", time.time()-start_time)

            display = num_episodes % display_freq == 0
            self.logger.flush(display)
            if display:
                self.tqf.display()
                self.display_run()

        self.logger.flush(True)
        print("Training complete")
        self.display_run()

    def run_episode(self):
        o = self.env.reset()
        done = False
        episode_return = 0
        while not done and self.steps_done < self.training_steps:
            a = self.get_action(o)
            # a = int(input("a:"))
            next_o, r, done, _ = self.env.step(a)
            # self.env.render()
            # print(f"o: {o}, a: {a}, o': {next_o}, r: {r:.6f}, d: {done}")
            # print(f"q_vals: {self.tqf(o)}")

            self.replay.store(o, a, next_o, r, done)
            o = next_o
            episode_return += r
            self.steps_done += 1

            if self.steps_done % self.function_update_freq == 0:
                mean_v, mean_td_error = self.optimize()

            if self.model_save_freq is not None and \
               self.steps_done % self.model_save_freq == 0:
                save_path = self.logger.get_save_path(ext=".yaml")
                self.tqf.save(save_path)

        self.return_tracker.update(episode_return)
        self.logger.log("episode_return", episode_return)
        self.logger.log("episode_mean_v", mean_v)
        self.logger.log("episode_mean_td_error", mean_td_error)
        self.logger.log("mean_episode_return", self.return_tracker.mean)
        self.logger.log("min_episode_return", self.return_tracker.min_val)
        self.logger.log("max_episode_return", self.return_tracker.max_val)
        self.logger.log("episode_return_stdev", self.return_tracker.stdev)

    def display_run(self):
        print("Running policy")
        o = self.env.reset()
        done = False
        episode_return = 0
        self.env.render()
        while not done:
            a = self.get_action(o)
            o, r, done, _ = self.env.step(a)
            episode_return += r
            self.env.render()
            input(f"Step = o: {o}, a: {a}, r: {r:.4f}, d: {done}")

        print("Episode done:")
        print(f"Reward = {episode_return}")
