import math
import pandas as pd
import matplotlib.pyplot as plt


NUM_COLS = 5


def plot_xy(ax, df, x_key, y_key, smooth):
    x = df[x_key]
    y = df[y_key]

    y_smooth = y.rolling(smooth, min_periods=1)
    y_smooth_mean = y_smooth.mean()
    y_rolling_std = y_smooth.std()

    ax.plot(x, y_smooth_mean)
    ax.fill_between(x, y_smooth_mean-y_rolling_std, y_smooth_mean+y_rolling_std, alpha=0.5)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)


def plot_all(file_path, x_key, smooth):
    df = pd.read_table(file_path)

    num_plots = len(df.columns)-1
    print(f"Generating {num_plots} plots")

    fig_cols = NUM_COLS
    fig_rows = math.ceil(num_plots / NUM_COLS)

    fig, axs = plt.subplots(fig_rows, fig_cols, sharex=True, figsize=(12, 10))

    plots_done = 0
    for row in range(fig_rows):
        for col in range(fig_cols):
            if plots_done > num_plots:
                break
            ax = axs[row][col]
            y_key = df.columns[plots_done]
            plot_xy(ax, df, x_key, y_key, smooth)
            plots_done += 1
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("--x_key", type=str, default="epoch")
    parser.add_argument("--smooth", type=int, default=10)
    args = parser.parse_args()

    plot_all(**vars(args))
