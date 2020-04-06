"""Plot all metrics for multiple seeds of the same experiment """
import math
import pandas as pd
import matplotlib.pyplot as plt


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


def plot_all(dfs, x_key, run_key, smooth):
    column_keys = list(dfs[0].columns)
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
            for df in dfs:
                plot_xy(ax, df, x_key, y_key, run_key, smooth)
            plots_done += 1
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_paths", type=str, nargs="*")
    parser.add_argument("--x_key", type=str, default="epoch",
                        help="Key to plot on X Axis (default='epoch')")
    parser.add_argument("--run_key", type=str, default=None,
                        help="Key to use to distinguish runs (default=None)")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Smoothing window size (default=1)")
    args = parser.parse_args()

    dfs = []
    for fp in args.file_paths:
        dfs.append(pd.read_table(fp))

    plot_all(dfs, args.x_key, args.run_key, args.smooth)
