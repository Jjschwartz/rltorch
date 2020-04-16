from rltorch.papers.DQN.dqn_agent import DQNAgent
from rltorch.papers.DQN.ddqn_agent import DDQNAgent
from rltorch.papers.DQN.hyperparams import AtariHyperparams


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("alg", type=str, help="The algorithm to run")
    parser.add_argument("env_name", type=str, help="The env to run")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="(default=0)")
    args = parser.parse_args()

    print(args)

    if args.alg == "dqn":
        AtariHyperparams.set_mode("dqn", seed=args.seed)
        agent_cls = DQNAgent
    elif args.alg == "ddqn":
        AtariHyperparams.set_mode("ddqn", seed=args.seed)
        agent_cls = DDQNAgent
    elif args.alg == "ddqn-tuned":
        AtariHyperparams.set_mode("ddqn-tuned", seed=args.seed)
        agent_cls = DDQNAgent
    else:
        raise NotImplementedError("Algorithm not supported")

    if args.test:
        AtariHyperparams.set_mode("testing", seed=args.seed)

    agent = agent_cls(args.env_name)
    agent.train()
