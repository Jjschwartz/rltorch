import pandas as pd
import matplotlib.pyplot as plt


def plot_xy(file_path, x_key, y_key, smooth):

    df = pd.read_table(file_path)
    x = df[x_key]
    y = df[y_key]

    y_smooth = y.rolling(smooth, min_periods=1)
    y_smooth_mean = y_smooth.mean()
    y_rolling_std = y_smooth.std()

    plt.plot(x, y_smooth_mean)
    plt.fill_between(x, y_smooth_mean-y_rolling_std, y_smooth_mean+y_rolling_std, alpha=0.5)
    plt.xlabel(x_key)
    plt.ylabel(y_key)

    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("--x_key", type=str, default="episode")
    parser.add_argument("--y_key", type=str, default="episode_return")
    parser.add_argument("--smooth", type=int, default=10)
    args = parser.parse_args()

    plot_xy(**vars(args))
