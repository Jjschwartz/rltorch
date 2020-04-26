import rltorch.algs.q_learning.base.run as run
from rltorch.algs.q_learning.DuelingDQN.agent import DuelingDQNAgent


if __name__ == "__main__":
    parser = run.get_target_deep_q_argparse()
    parser.add_argument("--dueling_sizes",
                        type=int,
                        nargs="*",
                        default=[64],
                        help=("Layers for dueling streams (on top of hidden"
                              " sizes) (default=[64])"))
    args = parser.parse_args()

    kwargs = vars(args)
    runs = args.runs
    run.run_agent(DuelingDQNAgent, kwargs, args.runs)
