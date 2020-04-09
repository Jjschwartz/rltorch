import time

from rltorch.algs.q_learning.DQN.agent import DQNAgent
from rltorch.utils.compile_util import move_dirs_into_single_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='CartPole-v0')
    parser.add_argument("--hidden_sizes", type=int, nargs="*",
                        default=[64, 64])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--training_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay_size", type=int, default=100000)
    parser.add_argument("--final_epsilon", type=float, default=0.01)
    parser.add_argument("--init_epsilon", type=float, default=1.0)
    parser.add_argument("--exploration", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--start_steps", type=int, default=32)
    parser.add_argument("--network_update_freq", type=int, default=1)
    parser.add_argument("--model_save_freq", type=int, default=400000)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    kwargs = vars(args)
    seed = args.seed
    save_dirs = []
    for i in range(args.runs):
        print(f"\n{'='*60}\nDQN Run {i}\n{'='*60}")
        kwargs["seed"] = seed + i
        agent = DQNAgent(**vars(args))
        agent.train()
        save_dirs.append(agent.logger.save_dir)

    if args.runs > 1:
        print("Moving results into a parent dir")
        ts = time.strftime("%Y%m%d")
        parent_dir_name = f"{args.env_name}_{args.runs}_runs_{ts}"
        if args.exp_name:
            parent_dir_name = f"{args.exp_name}_{parent_dir_name}"
        else:
            parent_dir_name = f"DQN_{parent_dir_name}"
        move_dirs_into_single_dir(save_dirs, parent_dir_name)
