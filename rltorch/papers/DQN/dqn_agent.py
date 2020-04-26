import gym
import time
import math
import random
import numpy as np
from pprint import pprint
from gym.wrappers import AtariPreprocessing

import torch
import torch.nn as nn
import torch.optim as optim

from .replay import ReplayMemory
from .preprocess import ImageHistory
from rltorch.utils.rl_logger import RLLogger
from rltorch.utils.stat_utils import StatTracker
from rltorch.papers.DQN.hyperparams import AtariHyperparams as hp

RENDER_PAUSE = 0.01


class DQNAgent:

    def __init__(self, env_name, death_ends_episode=True):
        assert "NoFrameskip-v4" in env_name
        print(f"\n{hp.ALGO} for Atari: {env_name}")
        config = hp.get_all_hyperparams()
        pprint(config)

        torch.manual_seed(hp.SEED)
        np.random.seed(hp.SEED)

        self.env_name = env_name
        self.env = AtariPreprocessing(gym.make(env_name),
                                      scale_obs=hp.NORMALIZE)
        self.num_actions = self.env.action_space.n

        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")
        print(f"Using device={self.device}")

        self.replay = ReplayMemory(hp.REPLAY_SIZE,
                                   hp.AGENT_HISTORY,
                                   hp.WIDTH,
                                   hp.HEIGHT,
                                   hp.REPLAY_S_DTYPE,
                                   self.device)
        self.replay.display_memory_usage()
        self.img_buffer = ImageHistory(hp.AGENT_HISTORY, (hp.HEIGHT, hp.WIDTH))

        self.logger = RLLogger(self.env_name, f"{hp.ALGO}_atari")
        self.logger.save_config(config)
        self.eval_logger = RLLogger(self.env_name, f"{hp.ALGO}_atari_eval")
        self.return_tracker = StatTracker()

        # Neural Network related attributes
        self.dqn = hp.MODEL(self.num_actions).to(self.device)
        self.target_dqn = hp.MODEL(self.num_actions).to(self.device)
        print(self.dqn)

        self.optimizer = optim.RMSprop(self.dqn.parameters(),
                                       lr=hp.LEARNING_RATE,
                                       momentum=hp.GRADIENT_MOMENTUM,
                                       eps=hp.MIN_SQUARED_GRADIENT)

        print(self.optimizer)
        self.loss_fn = nn.SmoothL1Loss()

        # Training related attributes
        self.epsilon_schedule = np.linspace(hp.INITIAL_EXPLORATION,
                                            hp.FINAL_EXPLORATION,
                                            hp.FINAL_EXPLORATION_FRAME)
        self.death_ends_episode = death_ends_episode
        self.steps_done = 0
        self.updates_done = 0

    def load_model(self, file_path):
        self.dqn.load_DQN(file_path, device=self.device)
        self.update_target_net()

    def get_action(self, x):
        return self.get_egreedy_action(x, self.get_epsilon())

    def get_epsilon(self):
        if self.steps_done < hp.REPLAY_START_SIZE:
            return 1.0
        if self.steps_done < hp.FINAL_EXPLORATION_FRAME:
            return self.epsilon_schedule[self.steps_done]
        return hp.FINAL_EXPLORATION

    def get_egreedy_action(self, x, epsilon):
        if random.random() > epsilon:
            x = torch.from_numpy(x).to(self.device)
            return self.dqn.get_action(x).cpu().item()
        return random.randint(0, self.num_actions-1)

    def get_action_and_value(self, x):
        x = torch.from_numpy(x).to(self.device)
        q_vals = self.dqn(x).cpu()
        a = q_vals.max(1)[1]
        return q_vals.data, a.item()

    def optimize(self):
        if self.steps_done % hp.NETWORK_UPDATE_FREQUENCY != 0:
            return None

        if self.steps_done < hp.REPLAY_START_SIZE:
            return 0, 0, 0, 0

        batch = self.replay.sample_batch(hp.MINIBATCH_SIZE)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        q_vals_raw = self.dqn(s_batch)
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()

        with torch.no_grad():
            target_q_val = self.target_dqn(next_s_batch).max(1)[0]
            target = r_batch + hp.DISCOUNT*(1-d_batch)*target_q_val

        loss = self.loss_fn(target, q_vals)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updates_done += 1

        if self.updates_done % hp.TARGET_NETWORK_UPDATE_FREQ == 0:
            self.update_target_net()

        mean_v = q_vals_raw.max(1)[0].mean().item()
        max_v = q_vals.max().item()
        mean_td_error = (target - q_vals).abs().mean().item()
        return loss.item(), mean_v, max_v, mean_td_error

    def update_target_net(self):
        print(f"step={self.steps_done}, updates={self.updates_done}:"
              " updating target_net")
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def save_model(self):
        print("saving model")
        save_path = self.logger.get_save_path(ext=".pth")
        self.dqn.save_DQN(save_path)

    def train(self):
        print("Starting training")
        training_start_time = time.time()
        self.steps_done = 0
        num_episodes = 0
        steps_since_eval = 0
        steps_remaining = hp.TRAINING_FRAMES
        while steps_remaining > 0:
            start_time = time.time()
            ep_return, ep_steps = self.run_episode(steps_remaining)
            num_episodes += 1
            steps_remaining -= ep_steps
            steps_since_eval += ep_steps

            self.return_tracker.update(ep_return)
            training_time = time.time()-training_start_time
            self.logger.log("episode", num_episodes)
            self.logger.log("steps_done", self.steps_done)
            self.logger.log("updates_done", self.updates_done)
            self.logger.log("epsilon", self.get_epsilon())
            self.logger.log("episode_return", ep_return)
            self.logger.log("episode_return_moving_mean",
                            self.return_tracker.moving_mean)
            self.logger.log("episode_return_moving_min",
                            self.return_tracker.moving_min)
            self.logger.log("episode_return_moving_max",
                            self.return_tracker.moving_max)
            self.logger.log("episode_return_moving_stdev",
                            self.return_tracker.moving_stdev)
            self.logger.log("episode_return_overall_max",
                            self.return_tracker.max_val)
            self.logger.log("episode_time", time.time()-start_time)
            self.logger.log("total_training_time", training_time)

            display = num_episodes % hp.LOG_DISPLAY_FREQ == 0
            self.logger.flush(display)

            if steps_since_eval >= hp.EVAL_FREQ or steps_remaining <= 0:
                eval_logger_kwargs = {"training_step": self.steps_done,
                                      "training_episode": num_episodes,
                                      "training_time": training_time}
                self.run_eval(eval_logger_kwargs=eval_logger_kwargs)
                steps_since_eval = 0

        self.logger.flush(True)
        print("Training complete")

    def run_episode(self, step_limit, eval_run=False, render=False):
        xs = self.init_episode()
        done = False

        steps = 0
        episode_return = 0
        losses, mean_values, mean_td_errors = [], [], []
        overall_max_v = -math.inf

        if render:
            self.env.render()

        while not done and steps < step_limit:
            if eval_run:
                a = self.get_egreedy_action(xs, hp.EVAL_EPSILON)
            else:
                a = self.get_action(xs)

            next_x, r, done, _ = self.env.step(a)
            self.img_buffer.push(next_x)
            next_xs = self.img_buffer.get()

            if render:
                self.env.render()
                time.sleep(RENDER_PAUSE)

            if not eval_run:
                self.steps_done += 1
                clipped_r = np.clip(r, *hp.R_CLIP)
                self.replay.store(xs, a, next_x, clipped_r, done)

                result = self.optimize()
                if result is not None:
                    loss, mean_v, max_v, mean_td_error = result
                    losses.append(loss)
                    mean_values.append(mean_v)
                    overall_max_v = max(overall_max_v, max_v)
                    mean_td_errors.append(mean_td_error)

                if self.steps_done % hp.MODEL_SAVE_FREQ == 0:
                    self.save_model()

            xs = next_xs
            episode_return += r
            steps += 1

        if not eval_run:
            losses = np.array(losses)
            self.logger.log("episode_mean_loss", losses.mean())
            self.logger.log("episode_loss_max", losses.max())
            self.logger.log("episode_loss_min", losses.min())
            self.logger.log("episode_mean_v", np.array(mean_values).mean())
            self.logger.log("episode_max_v", overall_max_v)
            self.logger.log("episode_mean_td_error",
                            np.array(mean_td_errors).mean())
        return episode_return, steps

    def run_eval(self, eval_logger_kwargs=None, render=False):
        print("RUNNING EVALUATION")
        eval_steps_remaining = hp.EVAL_STEPS
        eval_tracker = StatTracker()
        eval_start_time = time.time()
        while eval_steps_remaining > 0:
            ep_return, ep_steps = self.run_episode(eval_steps_remaining,
                                                   eval_run=True,
                                                   render=render)
            eval_steps_remaining -= ep_steps
            print(f"Episode Fin. Return={ep_return}, "
                  f"eval_steps_remaining={eval_steps_remaining}")
            if eval_steps_remaining > 0:
                eval_tracker.update(ep_return)

        if eval_logger_kwargs is None:
            eval_logger_kwargs = {}
        self.eval_logger.log("training_step",
                             eval_logger_kwargs.get("training_step", 0))
        self.eval_logger.log("training_episode",
                             eval_logger_kwargs.get("training_episode", 0))
        self.eval_logger.log("training_time",
                             eval_logger_kwargs.get("training_time", 0))
        self.eval_logger.log("num_eval_episode", eval_tracker.n)
        self.eval_logger.log("episode_return_mean", eval_tracker.mean)
        self.eval_logger.log("episode_return_min", eval_tracker.min_val)
        self.eval_logger.log("episode_return_max", eval_tracker.max_val)
        self.eval_logger.log("episode_return_stdev", eval_tracker.stdev)
        self.eval_logger.log("eval_time", time.time() - eval_start_time)

        print("EVALUATION RESULTS:")
        self.eval_logger.flush(True)

    def init_episode(self):
        """Resets game, performs noops and returns first processed state """
        x = self.env.reset()
        self.img_buffer.push(x)
        # fill buffer
        for _ in range(hp.AGENT_HISTORY-1):
            x, _, _, _ = self.env.step(0)
            self.img_buffer.push(x)
        return self.img_buffer.get()
