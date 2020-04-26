from rltorch.papers.DQN.dqn_agent import DQNAgent
from rltorch.papers.DQN.ddqn_agent import DDQNAgent
from rltorch.papers.DQN.duelingdqn_agent import DuelingDQNAgent
from rltorch.papers.DQN.hyperparams import AtariHyperparams


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("alg", type=str, help="The algorithm to run")
    parser.add_argument("env_name", type=str, help="The env to run")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="(default=0)")
    args = parser.parse_args()

    print(args)
    AtariHyperparams.set_seed(args.seed)
    if args.alg == "dqn":
        AtariHyperparams.set_mode("dqn")
        agent_cls = DQNAgent
    elif args.alg == "ddqn":
        AtariHyperparams.set_mode("ddqn")
        agent_cls = DDQNAgent
    elif args.alg == "ddqn-tuned":
        AtariHyperparams.set_mode("ddqn-tuned")
        agent_cls = DDQNAgent
    elif args.alg == "duelingdqn":
        AtariHyperparams.set_mode("duelingdqn")
        agent_cls = DuelingDQNAgent
    else:
        raise NotImplementedError("Algorithm not supported")

    if args.test:
        AtariHyperparams.set_mode("testing")

    if args.normalized:
        AtariHyperparams.set_mode("normalized")

    agent = agent_cls(args.env_name)
    agent.train()
