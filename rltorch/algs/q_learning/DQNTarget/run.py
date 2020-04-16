import rltorch.algs.q_learning.base.run as run
from rltorch.algs.q_learning.DQNTarget.agent import DQNTargetAgent


if __name__ == "__main__":
    parser = run.get_deep_q_argparse()
    parser.add_argument("--target_update_freq", type=int, default=1000)
    args = parser.parse_args()

    kwargs = vars(args)
    runs = args.runs
    run.run_agent(DQNTargetAgent, kwargs, args.runs)
