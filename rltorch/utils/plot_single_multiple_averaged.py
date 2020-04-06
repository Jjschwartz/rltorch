"""Plot all metrics for multiple seeds of the same experiment """
import pandas as pd
import matplotlib.pyplot as plt

from rltorch.utils.compile_util import list_files


NUM_COLS = 5


def plot_xy(ax, grouped_df, group_keys, x_key, y_key):
    x = group_keys
    y_mean = grouped_df.mean()[y_key]
    y_std = grouped_df.std()[y_key]

    print(f"Final {y_key} = {y_mean.iloc[-1]} +/- {y_std.iloc[-1]}")

    ax.plot(x, y_mean)
    ax.fill_between(x, y_mean-y_std, y_mean+y_std, alpha=0.5)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)


def plot_single(grouped_df, x_key, y_key):
    print(f"Generating 1 plot")
    group_keys = list(grouped_df.groups.keys())

    fig, ax = plt.subplots(1, 1)
    plot_xy(ax, grouped_df, group_keys, x_key, y_key)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=str, nargs="*", default=None,
                        help="List of file paths to results (default=None)")
    parser.add_argument("--dp", type=str, default=None,
                        help=("Parent dir of dirs containing results "
                              "(default=None)"))
    parser.add_argument("--y_key", type=str, default="avg_return",
                        help="Key to plot on Y Axis (default='avg_return')")
    parser.add_argument("--x_key", type=str, default="epoch",
                        help="Key to plot on X Axis (default='epoch')")
    parser.add_argument("--agg_key", type=str, default="epoch",
                        help="Key to aggregate runs over (default='epoch')")
    args = parser.parse_args()

    if args.fps is not None:
        results_fps = args.fps
    elif args.dp is not None:
        _, results_fps = list_files(args.dp)
    else:
        raise ValueError("Must supply either file paths of dir path")

    dfs = []
    for fp in results_fps:
        dfs.append(pd.read_table(fp))

    concat_df = pd.concat(dfs)
    grouped_df = concat_df.groupby(args.agg_key)

    plot_single(grouped_df, args.x_key, args.y_key)
