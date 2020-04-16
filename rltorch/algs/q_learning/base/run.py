import time
import argparse
import gym_maze

from rltorch.utils.compile_util import move_dirs_into_single_dir


def get_deep_q_argparse():
    parser = get_base_argparse()
    parser.add_argument("--hidden_sizes", type=int, nargs="*",
                        default=[64, 64])
    parser.add_argument("--network_update_freq", type=int, default=1)
    return parser


def get_base_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='CartPole-v0')
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default=0.001)")
    parser.add_argument("--training_steps", type=int, default=100000,
                        help="training steps (default=100000)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay_size", type=int, default=100000)
    parser.add_argument("--final_epsilon", type=float, default=0.05)
    parser.add_argument("--init_epsilon", type=float, default=1.0)
    parser.add_argument("--exploration", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--start_steps", type=int, default=32)
    parser.add_argument("--model_save_freq", type=int, default=400000)
    parser.add_argument("--eval_freq", type=int, default=100000)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--eval_epsilon", type=float, default=0.01)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--runs", type=int, default=1)
    return parser


def run_agent(agent_cls, agent_kwargs, num_runs=1):
    env_name = agent_kwargs["env_name"]
    save_dirs = []
    seed = agent_kwargs.get("seed", 0)
    for i in range(num_runs):
        print(f"\n{'='*60}\nRun {i}\n{'='*60}")
        agent_kwargs["seed"] = seed + i
        agent = agent_cls(**agent_kwargs)
        agent.train()
        save_dirs.append(agent.logger.save_dir)

    if num_runs > 1:
        print("Moving results into a parent dir")
        ts = time.strftime("%Y%m%d")
        parent_dir_name = f"{env_name}_{num_runs}_runs_{ts}"
        exp_name = agent_kwargs.get("exp_name", None)
        if not exp_name:
            exp_name = str(agent.__class__.__name__)
        parent_dir_name = f"{exp_name}_{parent_dir_name}"
        move_dirs_into_single_dir(save_dirs, parent_dir_name)
