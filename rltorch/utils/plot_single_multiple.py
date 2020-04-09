"""Plot all metrics for multiple seeds of the same experiment """
import pandas as pd
import matplotlib.pyplot as plt

from rltorch.utils.compile_util import list_files


NUM_COLS = 5


def plot_xy(ax, df, x_key, y_key, run_key, smooth):
    x = df[x_key]
    y = df[y_key]

    y_smooth = y.rolling(smooth, min_periods=1)
    y_smooth_mean = y_smooth.mean()
    y_rolling_std = y_smooth.std()

    if run_key is None:
        ax.plot(x, y_smooth_mean)
    else:
        ax.plot(x, y_smooth_mean, label=run_key)

    ax.fill_between(x,
                    y_smooth_mean-y_rolling_std,
                    y_smooth_mean+y_rolling_std,
                    alpha=0.5)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)


def plot_all(dfs, x_key, run_key, y_key, smooth):
    print(f"Generating 1 plot")
    fig, ax = plt.subplots(1, 1)

    for df in dfs:
        plot_xy(ax, df, x_key, y_key, run_key, smooth)
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
    parser.add_argument("--run_key", type=str, default=None,
                        help="Key to use to distinguish runs (default=None)")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Smoothing window size (default=1)")
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

    plot_all(dfs, args.x_key, args.run_key, args.y_key, args.smooth)
