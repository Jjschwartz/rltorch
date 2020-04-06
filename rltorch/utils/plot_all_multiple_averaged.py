"""Plot all metrics for multiple seeds of the same experiment """
import math
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


def plot_all(grouped_df, x_key):
    group_keys = list(grouped_df.groups.keys())
    column_keys = list(grouped_df.get_group(group_keys[0]).columns)
    column_keys.remove(x_key)
    num_plots = len(column_keys)-1
    print(f"Generating {num_plots} plots")

    fig_cols = min(num_plots, NUM_COLS)
    fig_rows = math.ceil(num_plots / fig_cols)

    fig, axs = plt.subplots(fig_rows, fig_cols, sharex=True, figsize=(12, 10),
                            squeeze=False)

    plots_done = 0
    for row in range(fig_rows):
        for col in range(fig_cols):
            if plots_done > num_plots:
                break
            ax = axs[row][col]
            y_key = column_keys[plots_done]
            plot_xy(ax, grouped_df, group_keys, x_key, y_key)
            plots_done += 1
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
    parser.add_argument("--x_key", type=str, default="epoch",
                        help="Key to plot on X Axis (default='epoch')")
    parser.add_argument("--agg_key", type=str, default="epoch",
                        help="Key to aggregate runs over (default='epoch')")
    args = parser.parse_args()

    if args.fps is not None:
        result_fps = args.fps
    elif args.dp is not None:
        _, results_fps = list_files(args.dp)
    else:
        raise ValueError("Must supply either file paths of dir path")

    dfs = []
    for fp in results_fps:
        dfs.append(pd.read_table(fp))

    concat_df = pd.concat(dfs)
    grouped_df = concat_df.groupby(args.agg_key)

    plot_all(grouped_df, args.x_key)
