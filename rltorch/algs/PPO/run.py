from rltorch.algs.PPO.agent import PPOAgent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='CartPole-v0')
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[32, 32])
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--epoch_steps", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--target_kl", type=float, default=0.01)
    parser.add_argument("--train_actor_iters", type=int, default=80)
    parser.add_argument("--train_critic_iters", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_save_freq", type=int, default=50)
    args = parser.parse_args()

    agent = PPOAgent(**vars(args))
    agent.train()
