import time
import random
import numpy as np
import os.path as osp
from gym.envs.atari import AtariEnv

import torch
import torch.nn as nn
import torch.optim as optim

from .model import DQN
from .replay import ReplayMemory
from .dqn_logger import DQNLogger, RESULTS_DIR
import rltorch.papers.DQN.hyperparams as hp
from .preprocess import ImageProcessor, ImageHistory

# TODO Look into storing tensors rather than numpy, for efficieny and to save
# repeated numpy -> tensor conversions. Maybe it's not a big deal.


class DQNAgent:

    def __init__(self, env_name, death_ends_episode=True):
        self.env_name = env_name
        self.env = AtariEnv(game=env_name, frameskip=1, obs_type="image")
        self.num_actions = self.env.action_space.n
        self.replay = ReplayMemory(hp.REPLAY_SIZE, hp.STATE_DIMS)
        self.img_processor = ImageProcessor(hp.HEIGHT, hp.WIDTH)
        self.img_buffer = ImageHistory(hp.AGENT_HISTORY, (hp.HEIGHT, hp.WIDTH))
        self.logger = DQNLogger(env_name)
        self.setup_logger()

        # Neural Network related attributes
        self.dqn = DQN(self.num_actions)
        self.target_dqn = DQN(self.num_actions)
        self.update_target_net()
        self.optimizer = optim.RMSprop(self.dqn.parameters(),
                                       lr=hp.LEARNING_RATE,
                                       momentum=hp.GRADIENT_MOMENTUM,
                                       eps=hp.MIN_SQUARED_GRADIENT)
        self.loss_fn = nn.MSELoss()

        # Training related attributes
        self.epsilon_schedule = np.linspace(hp.INITIAL_EXPLORATION, hp.FINAL_EXPLORATION,
                                            hp.FINAL_EXPLORATION_FRAME)
        self.death_ends_episode = death_ends_episode
        self.steps_done = 0

    def setup_logger(self):
        # adds headers of interest
        self.logger.add_header("episode")
        self.logger.add_header("steps_done")
        self.logger.add_header("episode_return")
        self.logger.add_header("episode_loss")

    def get_model_save_path(self):
        ts = time.strftime("%Y%m%d-%H%M")
        return osp.join(RESULTS_DIR, f"{self.env_name}_{ts}.pth")

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
            return 0

        batch = self.replay.sample_batch(hp.MINIBATCH_SIZE)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        # get q_vals for each state and the action performed in that state
        q_vals_raw = self.dqn(s_batch)
        a_batch_tensor = torch.from_numpy(a_batch.reshape(32, 1))
        q_vals = q_vals_raw.gather(1, a_batch_tensor).squeeze()

        # get target q val = max val of next state
        _, target_q_val = self.target_dqn(next_s_batch).max(1)

        r_batch_tensor = torch.from_numpy(r_batch)
        d_batch_tensor = torch.from_numpy((1-d_batch))

        # calculate update target
        target = r_batch_tensor + hp.DISCOUNT*d_batch_tensor*target_q_val
        # calculate mean square loss
        loss = self.loss_fn(q_vals, target)
        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
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

        while self.steps_done < hp.TRAINING_FRAMES:
            ep_return, ep_loss = self.run_episode()
            episode_returns.append(ep_return)
            num_episodes += 1

            if num_episodes % 1 == 0:
                print(f"Episode {num_episodes}: return={ep_return:.4f} loss={ep_loss:.4f}"
                      f"\t ({self.steps_done} / {hp.TRAINING_FRAMES} steps complete)")

            self.logger.log("episode", num_episodes)
            self.logger.log("steps_done", self.steps_done)
            self.logger.log("episode_return", ep_return)
            self.logger.log("episode_loss", ep_loss)
            self.logger.flush()

        print("Training complete")

    def run_episode(self):
        s = self.init_episode()
        done = False
        start_lives = self.env.ale.lives()

        episode_return = 0
        episode_loss = 0

        while not done and self.steps_done < hp.TRAINING_FRAMES:
            self.img_buffer.push(s)
            x = self.img_buffer.get()
            a = self.get_action(x)
            next_s, r = self.step(a)

            life_lost = self.env.ale.lives() < start_lives
            if self.env.ale.game_over() or (self.death_ends_episode and life_lost):
                done = True

            clipped_r = np.clip(r, *hp.R_CLIP)
            self.replay.store(s, a, next_s, clipped_r, done)
            s = next_s
            episode_return += r
            self.steps_done += 1

            if self.steps_done % hp.NETWORK_UPDATE_FREQUENCY == 0:
                episode_loss = self.optimize()

            if self.steps_done % hp.TARGET_NETWORK_UPDATE_FREQ == 0:
                self.update_target_net()

            if self.steps_done % hp.MODEL_SAVE_FREQ == 0:
                save_path = self.get_model_save_path()
                self.dqn.save_DQN(save_path)

        return episode_return, episode_loss

    def step(self, a):
        """Perform a step, repeating given action ACTION_REPEAT times, and
        return processed image and reward """
        reward = 0.0
        img_buffer = []
        for i in range(hp.ACTION_REPEAT):
            s, r, d, info = self.env.step(a)
            reward += r
            img_buffer.append(s)
        x = self.img_processor.process_frames(*img_buffer[:-2])
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
        return x
