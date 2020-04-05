from rltorch.algs.DDQN.agent import DDQNAgent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='CartPole-v0')
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[16, 8])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--training_steps", type=int, default=300000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay_size", type=int, default=50000)
    parser.add_argument("--final_epsilon", type=float, default=0.01)
    parser.add_argument("--init_epsilon", type=float, default=1.0)
    parser.add_argument("--exploration", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--start_steps", type=int, default=32)
    parser.add_argument("--network_update_freq", type=int, default=1)
    parser.add_argument("--target_update_freq", type=int, default=1000)
    parser.add_argument("--model_save_freq", type=int, default=400000)
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    agent = DDQNAgent(**vars(args))
    agent.train()
