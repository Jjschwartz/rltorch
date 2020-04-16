import gym
import time
import math
import random
import numpy as np
from pprint import pprint


from rltorch.utils.rl_logger import RLLogger
from rltorch.utils.stat_utils import StatTracker

PAUSE_DISPLAY = False
DISPLAY_DELAY = 0.01


class QLearningBaseAgent:

    def __init__(self, alg_name, **kwargs):
        print(f"\nRunning {alg_name} with config:")
        pprint(kwargs)

        # set seeds
        self.seed = kwargs["seed"]
        if self.seed is not None:
            np.random.seed(self.seed)

        # envirnment setup
        self.env_name = kwargs["env_name"]
        self.env = gym.make(self.env_name)
        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape

        # Logger setup
        logger_name = alg_name
        if "exp_name" in kwargs and kwargs["exp_name"]:
            logger_name = kwargs["exp_name"]
        self.logger = RLLogger(self.env_name, logger_name)
        self.eval_logger = RLLogger(self.env_name, f"{logger_name}_eval")
        self.logger.save_config(kwargs)
        self.setup_logger()
        self.return_tracker = StatTracker()

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
        self.steps_done = 0
        self.updates_done = 0

        # Evaluation related attributes
        self.eval_freq = kwargs["eval_freq"]
        self.eval_steps = kwargs["eval_steps"]
        self.eval_epsilon = kwargs["eval_epsilon"]

        # other attributes
        self.model_save_freq = kwargs["model_save_freq"]

    def setup_logger(self):
        self.logger.add_header("episode")
        self.logger.add_header("seed")
        self.logger.add_header("steps_done")
        self.logger.add_header("updates_done")
        self.logger.add_header("epsilon")
        self.logger.add_header("episode_return")
        self.logger.add_header("episode_loss")
        self.logger.add_header("episode_mean_v")
        self.logger.add_header("episode_max_v")
        self.logger.add_header("episode_mean_td_error")
        self.logger.add_header("episode_return_moving_mean")
        self.logger.add_header("episode_return_moving_min")
        self.logger.add_header("episode_return_moving_max")
        self.logger.add_header("episode_return_moving_stdev")
        self.logger.add_header("episode_return_overall_max")
        self.logger.add_header("episode_time")
        self.logger.add_header("total_training_time")

        self.eval_logger.add_header("training_step")
        self.eval_logger.add_header("training_episode")
        self.eval_logger.add_header("training_time")
        self.eval_logger.add_header("num_eval_episode")
        self.eval_logger.add_header("episode_return_mean")
        self.eval_logger.add_header("episode_return_min")
        self.eval_logger.add_header("episode_return_max")
        self.eval_logger.add_header("episode_return_stdev")
        self.eval_logger.add_header("eval_time")

    def get_epsilon(self):
        if self.steps_done < self.start_steps:
            return 1.0
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        return self.final_epsilon

    def get_training_action(self, o):
        return self.get_egreedy_action(o, self.get_epsilon())

    def get_eval_action(self, o):
        return self.get_egreedy_action(o, self.eval_epsilon)

    def get_egreedy_action(self, o, epsilon):
        if random.random() > epsilon:
            return self.get_action(o)
        return random.randint(0, self.num_actions-1)

    def get_action(self, o):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def store(self, o, a, next_o, r, d):
        raise NotImplementedError

    def train(self):
        print("Starting training")
        training_start_time = time.time()
        num_episodes = 0
        display_freq = min(100, int(self.training_steps // 10))
        steps_since_eval = 0
        training_steps_remaining = self.training_steps
        while self.steps_done < self.training_steps:
            start_time = time.time()
            ep_return, ep_steps = self.run_episode(training_steps_remaining)
            num_episodes += 1
            steps_since_eval += ep_steps
            training_steps_remaining -= ep_steps

            training_time = time.time()-training_start_time
            self.logger.log("episode", num_episodes)
            self.logger.log("steps_done", self.steps_done)
            self.logger.log("updates_done", self.updates_done)
            self.logger.log("epsilon", self.get_epsilon())
            self.logger.log("seed", self.seed)
            self.log_training_episode_return(ep_return)
            self.logger.log("episode_time", time.time()-start_time)
            self.logger.log("total_training_time", training_time)

            display = num_episodes % display_freq == 0
            self.logger.flush(display)

            if steps_since_eval >= self.eval_freq or \
               training_steps_remaining <= 0:
                # render last episode
                render = training_steps_remaining <= 0
                eval_logger_kwargs = {"training_step": self.steps_done,
                                      "training_episode": num_episodes,
                                      "training_time": training_time}
                self.run_eval(eval_logger_kwargs=eval_logger_kwargs,
                              render=render)
                steps_since_eval = 0

        self.logger.flush(True)
        print("Training complete")

    def log_training_episode_return(self, ep_return):
        self.return_tracker.update(ep_return)
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

    def run_episode(self, step_limit, eval_run=False, render=False):
        o = self.env.reset()
        done = False

        steps = 0
        episode_return = 0
        losses, mean_values, mean_td_errors = [], [], []
        overall_max_v = -math.inf

        if render:
            self.env.render()
            time.sleep(DISPLAY_DELAY)

        while not done and steps < step_limit:
            if eval_run:
                a = self.get_eval_action(o)
            else:
                a = self.get_training_action(o)
            next_o, r, done, _ = self.env.step(a)

            if render:
                self.env.render()
                time.sleep(DISPLAY_DELAY)

            if not eval_run:
                self.store(o, a, next_o, r, done)
                self.steps_done += 1

                result = self.optimize()
                if result is not None:
                    loss, mean_v, max_v, mean_td_error = result
                    losses.append(loss)
                    mean_values.append(mean_v)
                    overall_max_v = max(overall_max_v, max_v)
                    mean_td_errors.append(mean_td_error)

                if self.model_save_freq is not None and \
                   self.steps_done % self.model_save_freq == 0:
                    self.save_model()

            o = next_o
            episode_return += r
            steps += 1

        if not eval_run:
            self.logger.log("episode_loss", np.array(losses).mean())
            self.logger.log("episode_mean_v", np.array(mean_values).mean())
            self.logger.log("episode_max_v", overall_max_v)
            self.logger.log("episode_mean_td_error",
                            np.array(mean_td_errors).mean())

        return episode_return, steps

    def run_eval(self, eval_logger_kwargs=None, render=False):
        print("RUNNING EVALUATION")
        eval_steps_remaining = self.eval_steps
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
