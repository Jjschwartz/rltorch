import rltorch.algs.q_learning.base.run as run
from rltorch.algs.q_learning.DQN.agent import DQNAgent


if __name__ == "__main__":
    parser = run.get_deep_q_argparse()
    args = parser.parse_args()

    kwargs = vars(args)
    runs = args.runs
    run.run_agent(DQNAgent, kwargs, args.runs)
