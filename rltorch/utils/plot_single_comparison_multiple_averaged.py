"""Plot a single metric for multiple experiments with each averages over
multiple seeds
"""
import matplotlib.pyplot as plt

from rltorch.utils import file_utils as fu
from rltorch.utils import plot_single_multiple_averaged as pu


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dps", type=str, nargs="*",
                        help=("List of Parent dir of dirs containing results "
                              "REQUIRED"))
    parser.add_argument("--y_key", type=str, default="episode_return",
                        help=("Key to plot on Y Axis "
                              "(default='episode_return')"))
    parser.add_argument("--x_key", type=str, default="episode",
                        help="Key to plot on X Axis (default='episode')")
    parser.add_argument("--agg_key", type=str, default=None,
                        help=("Key to aggregate runs over, if None uses x_key"
                              " (default=None)"))
    args = parser.parse_args()

    fig, ax = plt.subplots(1, 1)

    for dp in args.dps:
        dir_name = fu.get_dir_name(dp)
        label = dir_name.split("_")[0]
        pu.plot_from_dir(ax, dp, args.x_key, args.y_key, args.agg_key, label)

    ax.legend()
    fig.tight_layout()
    plt.show()
