import time
import random
import numpy as np
from pprint import pprint
from gym.envs.atari import AtariEnv

import torch
import torch.nn as nn
import torch.optim as optim

from .model import DQN
from .replay import ReplayMemory
import rltorch.papers.DQN.hyperparams as hp
from rltorch.utils.rl_logger import RLLogger
from rltorch.utils.stat_utils import StatTracker
from .preprocess import ImageProcessor, ImageHistory

RENDER = False


class DQNAgent:

    def __init__(self, env_name, death_ends_episode=True):
        print("\nDQN for Atari: {env_name}")
        pprint(hp.ALL_KWARGS)

        self.env_name = env_name
        self.env = AtariEnv(game=env_name, frameskip=1, obs_type="image")
        self.num_actions = self.env.action_space.n

        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")
        print(f"Using device={self.device}")

        self.replay = ReplayMemory(hp.REPLAY_SIZE,
                                   hp.AGENT_HISTORY,
                                   hp.WIDTH,
                                   hp.HEIGHT,
                                   self.device)
        self.replay.display_memory_usage()
        self.img_processor = ImageProcessor(hp.HEIGHT, hp.WIDTH)
        self.img_buffer = ImageHistory(hp.AGENT_HISTORY, (hp.HEIGHT, hp.WIDTH))

        logger_name = "dqn_atari"
        self.logger = RLLogger(self.env_name, logger_name)
        self.setup_logger()
        self.logger.save_config(hp.ALL_KWARGS)
        self.stat_tracker = StatTracker()

        # Neural Network related attributes
        self.dqn = DQN(self.num_actions).to(self.device)
        self.target_dqn = DQN(self.num_actions).to(self.device)
        self.optimizer = optim.RMSprop(self.dqn.parameters(),
                                       lr=hp.LEARNING_RATE,
                                       alpha=hp.GRADIENT_MOMENTUM,
                                       eps=hp.MIN_SQUARED_GRADIENT)
        self.loss_fn = nn.MSELoss()

        # Training related attributes
        self.epsilon_schedule = np.linspace(hp.INITIAL_EXPLORATION,
                                            hp.FINAL_EXPLORATION,
                                            hp.FINAL_EXPLORATION_FRAME)
        self.death_ends_episode = death_ends_episode
        self.steps_done = 0

    def setup_logger(self):
        self.logger.add_header("episode")
        self.logger.add_header("steps_done")
        self.logger.add_header("episode_return")
        self.logger.add_header("episode_loss")
        self.logger.add_header("cumulative_return")
        self.logger.add_header("mean_episode_return")
        self.logger.add_header("min_episode_return")
        self.logger.add_header("max_episode_return")
        self.logger.add_header("episode_return_stdev")
        self.logger.add_header("episode_time")
        self.logger.add_header("total_training_time")

    def get_action(self, x):
        if self.steps_done < hp.REPLAY_START_SIZE:
            return random.randint(0, self.num_actions-1)

        if self.steps_done < hp.FINAL_EXPLORATION_FRAME:
            epsilon = self.epsilon_schedule[self.steps_done]
        else:
            epsilon = hp.FINAL_EXPLORATION

        if random.random() > epsilon:
            x = torch.from_numpy(x).to(self.device)
            return self.dqn.get_action(x).cpu().numpy()
        return random.randint(0, self.num_actions-1)

    def optimize(self):
        if self.steps_done < hp.REPLAY_START_SIZE:
            return 0

        batch = self.replay.sample_batch(hp.MINIBATCH_SIZE)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        q_vals_raw = self.dqn(s_batch)
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()
        target_q_val, _ = self.target_dqn(next_s_batch).max(1)
        target = r_batch + hp.DISCOUNT*(1-d_batch)*target_q_val
        loss = self.loss_fn(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn.parameters():
            # clip squared gradient
            param.grad.data.clamp_(*hp.GRAD_CLIP)
        self.optimizer.step()

        loss_value = loss.item()
        return loss_value

    def update_target_net(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def train(self):
        print("Starting training")
        training_start_time = time.time()
        self.steps_done = 0
        num_episodes = 0
        while self.steps_done < hp.TRAINING_FRAMES:
            start_time = time.time()
            ep_return, ep_loss = self.run_episode()
            num_episodes += 1

            self.stat_tracker.update(ep_return)

            self.logger.log("episode", num_episodes)
            self.logger.log("steps_done", self.steps_done)
            self.logger.log("episode_return", ep_return)
            self.logger.log("episode_loss", ep_loss)
            self.logger.log("cumulative_return", self.stat_tracker.total)
            self.logger.log("mean_episode_return", self.stat_tracker.mean)
            self.logger.log("min_episode_return", self.stat_tracker.min_val)
            self.logger.log("max_episode_return", self.stat_tracker.max_val)
            self.logger.log("episode_return_stdev", self.stat_tracker.stdev)
            self.logger.log("episode_time", time.time()-start_time)
            self.logger.log("total_training_time",
                            time.time()-training_start_time)

            display = num_episodes % 10 == 0
            self.logger.flush(display)

        self.logger.flush(True)
        print("Training complete")

    def run_episode(self):
        xs = self.init_episode()
        done = False
        start_lives = self.env.ale.lives()

        episode_return = 0
        episode_loss = 0

        while not done and self.steps_done < hp.TRAINING_FRAMES:
            if RENDER:
                self.env.render()

            a = self.get_action(xs)
            next_x, r = self.step(a)
            self.img_buffer.push(next_x)
            next_xs = self.img_buffer.get()

            life_lost = self.env.ale.lives() < start_lives
            done = (self.env.ale.game_over() or
                    (self.death_ends_episode and life_lost))

            clipped_r = np.clip(r, *hp.R_CLIP)
            self.replay.store(xs, a, next_x, clipped_r, done)
            xs = next_xs
            episode_return += r
            self.steps_done += 1

            if self.steps_done % hp.NETWORK_UPDATE_FREQUENCY == 0:
                episode_loss = self.optimize()

            if self.steps_done % hp.TARGET_NETWORK_UPDATE_FREQ == 0:
                self.update_target_net()

            if self.steps_done % hp.MODEL_SAVE_FREQ == 0:
                save_path = self.logger.get_save_path(ext=".pth")
                self.dqn.save_DQN(save_path)

        return episode_return, episode_loss

    def step(self, a):
        """Perform a step, repeating given action ACTION_REPEAT times, and
        return processed image and reward """
        reward = 0.0
        tmp_buffer = []
        for i in range(hp.ACTION_REPEAT):
            s, r, d, info = self.env.step(a)
            reward += r
            tmp_buffer.append(s)
        x = self.img_processor.process_frames(*tmp_buffer[:-2])
        return x, reward

    def init_episode(self):
        """Resets game, performs noops and returns first processed state """
        self.env.reset()
        self.img_buffer.clear()
        num_noops = random.randint(0, hp.NO_OP_MAX)
        for _ in range(num_noops):
            self.env.step(0)
        # ensure history buffer is full
        for _ in range(hp.AGENT_HISTORY):
            x, _ = self.step(0)
            self.img_buffer.push(x)
        return self.img_buffer.get()
