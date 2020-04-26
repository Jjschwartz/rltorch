"""Plot all metrics for multiple seeds of the same experiment """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rltorch.utils.compile_util import list_files


def plot_xy(ax, grouped_df, group_key, x_key, y_key, label=None):
    x = group_key
    y_mean = grouped_df.mean()[y_key]
    y_std = grouped_df.std()[y_key]

    print(f"{label} Final {y_key} = {y_mean.iloc[-1]} +/- {y_std.iloc[-1]}")

    if len(x) > 1:
        ax.plot(x, y_mean, label=label)
        ax.fill_between(x, y_mean-y_std, y_mean+y_std, alpha=0.25)
    else:
        ax.errorbar(x, y_mean, yerr=(y_mean-y_std, y_mean+y_std), fmt='o',
                    label=label, capsize=10)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)


def plot_single(ax, grouped_df, x_key, y_key, label=None):
    group_keys = list(grouped_df.groups.keys())
    plot_xy(ax, grouped_df, group_keys, x_key, y_key, label)


def plot_multiple_files(ax, file_paths, x_key, y_key, agg_key=None,
                        label=None):
    dfs = []
    for fp in file_paths:
        dfs.append(pd.read_table(fp))

    if agg_key is None:
        agg_key = x_key

    if agg_key == 'steps_done':
        max_steps = dfs[0][agg_key].max()
        num_bins = int(max_steps // 500)
        custom_bins = np.linspace(0, max_steps+1, num_bins, dtype=np.int)
        for i in range(len(dfs)):
            df = dfs[i]
            step_groups = pd.cut(df[agg_key], bins=custom_bins,
                                 labels=custom_bins[1:])

            df.insert(0, "steps", step_groups)
            df.drop_duplicates("steps", keep='last', inplace=True)
        agg_key = "steps"
        x_key = "steps"

    concat_df = pd.concat(dfs)
    grouped_df = concat_df.groupby(agg_key)
    print(concat_df)

    plot_single(ax, grouped_df, x_key, y_key, label)


def plot_from_dir(ax, parent_dir, x_key, y_key, agg_key=None, label=None):
    _, file_paths = list_files(parent_dir)
    plot_multiple_files(ax, file_paths, x_key, y_key, agg_key, label)


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

    fig, ax = plt.subplots(1, 1)

    if args.fps is not None:
        plot_multiple_files(ax, args.fps, args.x_key, args.y_key, args.agg_key)
    elif args.dp is not None:
        _, fps = list_files(args.dp)
        plot_multiple_files(ax, fps, args.x_key, args.y_key, args.agg_key)
    else:
        raise ValueError("Must supply either file paths of dir path")

    fig.tight_layout()
    plt.show()
