import rltorch.utils.file_utils as futils
from rltorch.papers.DQN.dqn_agent import DQNAgent
from rltorch.papers.DQN.hyperparams import AtariHyperparams


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    config = futils.load_yaml(args.config)
    AtariHyperparams.set_mode("eval")
    agent = DQNAgent(args.env_name)
    agent.load_model(args.model)
    agent.run_eval(render=True)
