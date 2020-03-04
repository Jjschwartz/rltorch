import gym
import time
import numpy as np
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim

from .buffer import PPOBuffer
from .model import PPOLSTMActorCritic
from rltorch.utils.rl_logger import RLLogger


class PPOLSTMAgent:

    def __init__(self, **kwargs):
        print("\nPPO with config:")
        pprint(kwargs)

        self.seed = kwargs["seed"]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.env_name = kwargs["env_name"]
        self.env = gym.make(self.env_name)
        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device={self.device}")

        self.logger = RLLogger(self.env_name, "ppolstm")
        self.setup_logger()

        # Hyper params
        self.steps_per_epoch = kwargs["epoch_steps"]
        self.epochs = kwargs["epochs"]
        self.max_ep_len = kwargs["max_ep_len"]
        self.clip_ratio = kwargs["clip_ratio"]
        self.target_kl = kwargs["target_kl"]
        self.train_actor_iters = kwargs["train_actor_iters"]
        self.train_critic_iters = kwargs["train_critic_iters"]
        self.model_save_freq = kwargs["model_save_freq"]

        self.buffer = PPOBuffer(self.steps_per_epoch, self.obs_dim, kwargs["hidden_size"],
                                kwargs["gamma"], kwargs["lam"], self.device)
        self.actor_critic = PPOLSTMActorCritic(self.obs_dim[0], kwargs["hidden_size"], self.num_actions)
        self.actor_critic.to(self.device)

        print("\nActorCritic:")
        print(self.actor_critic)

        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=kwargs["actor_lr"])
        self.critic_optimizer = optim.Adam(self.actor_critic.critic.parameters(), lr=kwargs["critic_lr"])
        self.critic_loss_fn = nn.MSELoss()

    def setup_logger(self):
        # adds headers of interest
        self.logger.add_header("epoch")
        self.logger.add_header("total_steps")
        self.logger.add_header("avg_ep_return")
        self.logger.add_header("min_ep_return")
        self.logger.add_header("max_ep_return")
        self.logger.add_header("avg_vals")
        self.logger.add_header("min_vals")
        self.logger.add_header("max_vals")
        self.logger.add_header("avg_ep_len")
        self.logger.add_header("actor_loss")
        self.logger.add_header("actor_loss_delta")
        self.logger.add_header("critic_loss")
        self.logger.add_header("critic_loss_delta")
        self.logger.add_header("kl")
        self.logger.add_header("entropy")
        self.logger.add_header("time")

    def get_action(self, obs, hid):
        return self.actor_critic.act(obs, hid)

    def compute_actor_loss(self, data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]
        hid = data["actor_hid"]
        pi, logp, _ = self.actor_critic.actor.step(obs, act, hid)
        ratio = torch.exp(logp - logp_old)
        clipped_ratio = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)
        clip_adv = clipped_ratio * adv
        actor_loss = -(torch.min(ratio * adv, clip_adv)).mean()

        actor_loss_info = dict()
        actor_loss_info["kl"] = (logp_old - logp).mean().item()
        actor_loss_info["entropy"] = pi.entropy().mean().item()
        return actor_loss, actor_loss_info

    def compute_critic_loss(self, data):
        obs, ret, hid = data["obs"], data["ret"], data["critic_hid"]
        predicted_val, _ = self.actor_critic.critic.forward(obs, hid)
        return self.critic_loss_fn(predicted_val, ret)

    def optimize(self):
        data = self.buffer.get()

        actor_loss_start, actor_loss_info_start = self.compute_actor_loss(data)
        actor_loss_start = actor_loss_start.item()
        critic_loss_start = self.compute_critic_loss(data).item()

        for i in range(self.train_actor_iters):
            self.actor_optimizer.zero_grad()
            actor_loss, actor_loss_info = self.compute_actor_loss(data)
            if actor_loss_info["kl"] > 1.5*self.target_kl:
                break
            actor_loss.backward()
            self.actor_optimizer.step()

        for i in range(self.train_critic_iters):
            self.critic_optimizer.zero_grad()
            critic_loss = self.compute_critic_loss(data)
            critic_loss.backward()
            self.critic_optimizer.step()

        # calculate changes in loss, for logging
        actor_loss_delta = (actor_loss.item() - actor_loss_start)
        critic_loss_delta = (critic_loss.item() - critic_loss_start)

        self.logger.log("actor_loss", actor_loss_start)
        self.logger.log("actor_loss_delta", actor_loss_delta)
        self.logger.log("critic_loss", critic_loss_start)
        self.logger.log("critic_loss_delta", critic_loss_delta)
        self.logger.log("kl", actor_loss_info_start["kl"])
        self.logger.log("entropy", actor_loss_info_start["entropy"])

    def step(self, obs, actor_hid, critic_hid):
        return self.actor_critic.step(self.process_single_obs(obs), actor_hid, critic_hid)

    def get_value(self, obs, critic_hid):
        return self.actor_critic.get_value(self.process_single_obs(obs), critic_hid)

    def process_single_obs(self, obs):
        proc_obs = torch.from_numpy(obs).float().to(self.device)
        proc_obs = proc_obs.view(1, 1, -1)
        return proc_obs

    def train(self):
        print("PPO Starting training")

        start_time = time.time()

        for epoch in range(self.epochs):
            self.logger.log("epoch", epoch)

            o = self.env.reset()
            epoch_ep_rets = []
            epoch_ep_lens = []
            ep_ret, ep_len = 0, 0
            epoch_vals = []

            actor_hid, critic_hid = self.actor_critic.get_init_hidden()

            for t in range(self.steps_per_epoch):
                a, v, logp, next_actor_hid, next_critic_hid = self.step(o, actor_hid, critic_hid)
                next_o, r, d, _ = self.env.step(a.squeeze())

                ep_len += 1
                ep_ret += r
                epoch_vals.append(v)
                self.buffer.store(o, a, r, v, logp, actor_hid, critic_hid)
                o = next_o
                actor_hid = next_actor_hid
                critic_hid = next_critic_hid

                timeout = ep_len == self.max_ep_len
                terminal = timeout or d
                epoch_ended = t == self.steps_per_epoch-1

                if terminal or epoch_ended:
                    v = 0
                    if timeout or epoch_ended:
                        v, next_critic_hid = self.get_value(o, critic_hid)
                    self.buffer.finish_path(v)

                if terminal:
                    epoch_ep_rets.append(ep_ret)
                    epoch_ep_lens.append(ep_len)
                    ep_ret, ep_len = 0, 0
                    o = self.env.reset()
                    actor_hid, critic_hid = self.actor_critic.get_init_hidden()

            # update the model
            self.optimize()

            # save model
            if (epoch+1) % self.model_save_freq == 0:
                print(f"Epoch {epoch+1}: saving model")
                save_path = self.logger.get_save_path("pth")
                self.actor_critic.save_AC(save_path)

            self.logger.log("total_steps", (epoch+1)*self.steps_per_epoch)
            self.logger.log("avg_ep_return", np.mean(epoch_ep_rets))
            self.logger.log("min_ep_return", np.min(epoch_ep_rets))
            self.logger.log("max_ep_return", np.max(epoch_ep_rets))
            self.logger.log("avg_vals", np.mean(epoch_vals))
            self.logger.log("min_vals", np.min(epoch_vals))
            self.logger.log("max_vals", np.max(epoch_vals))
            self.logger.log("avg_ep_len", np.mean(epoch_ep_lens))
            self.logger.log("time", time.time()-start_time)
            self.logger.flush(display=True)

        print("PPO Training complete")
