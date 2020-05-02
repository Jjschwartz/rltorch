"""Plot results """
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple

import rltorch.utils.plot_utils as pu
from rltorch.utils.compile_util import list_files


NUM_COLS = 5

Result = namedtuple('Result', ['label', 'x', 'ys'])


def average_results(results, x_key, y_keys, resample=512):
    """Average over dataframes

    Parameters
    ----------
    results : list[Result]
        list of result nametuples
    x_key : str
        result key to use for X-axis
    y_keys : list[str]
        results keys to use for Y-axis
    resample : int, optional
        number of datapoints to average over (default=512)

    Returns
    -------
    Result
       result namedtuble containint averaged results
    """
    label = results[0].label
    low = max(xys.x[0] for xys in results)
    high = min(xys.x[-1] for xys in results)
    x_new = np.linspace(low, high, resample)
    ys_new = {}
    for y_key in y_keys:
        yk_news = []
        for xys in results:
            x_old = xys.x
            y_old = xys.ys[y_key]
            yn = pu.symmetric_ema(x_old, y_old, low, high, resample)[1]
            yk_news.append(yn)
        ys_new[y_key] = yk_news
    return Result(label, x_new, ys_new)


def load_results(grouped_result_paths, x_key, y_keys, average, resample=512):
    """Load results from filepaths

    Parameters
    ----------
    grouped_result_paths : list
        of list of file paths where each list contains related filepaths
        to related results
    x_key : str
        key for X-axis
    y_keys : list
        of Y-Axis keys
    average : bool
        whether to average results or not
    resample : int, optional
        number of datapoints to average over (default=512)

    Returns
    -------
    all_results : list
        of Result namedtuples
    """
    plot_all_y = 'all' in y_keys

    all_results = []
    for rpaths in grouped_result_paths:
        group_results = []
        for i, fp in enumerate(rpaths):
            label = fp.split("_")[0]
            if not average and len(rpaths) > 1:
                label += f"_{i}"
            df = pd.read_table(fp)
            x = df[x_key].to_numpy()
            ys = {}
            if plot_all_y:
                y_keys = list(df.columns)
                y_keys.remove(x_key)
            for y_key in y_keys:
                ys[y_key] = df[y_key].to_numpy()
            group_results.append(Result(label, x, ys))
        if average:
            # average over each data frame
            avg_result = average_results(
                group_results, x_key, y_keys, resample
            )
            all_results.append(avg_result)
        else:
            all_results.extend(group_results)
    return all_results


def plot_results(all_results, x_key, y_keys):
    if 'all' in y_keys:
        print("Plotting results for all result metrics")
        y_keys = set()
        for result in all_results:
            y_keys.update(result.ys.keys())
        y_keys = list(y_keys)

    num_plots = len(y_keys)
    print(f"Generating {num_plots} plots")
    print(f"For {len(all_results)} results")
    fig_cols = min(num_plots, NUM_COLS)
    fig_rows = math.ceil(num_plots / fig_cols)

    fig, axs = plt.subplots(
        fig_rows, fig_cols, sharex=True, figsize=(12, 10), squeeze=False
    )

    plots_done = 0
    legend_handles = None
    for row in range(fig_rows):
        for col in range(fig_cols):
            if plots_done >= num_plots:
                break
            ax = axs[row][col]
            y_key = y_keys[plots_done]
            for result in all_results:
                if y_key not in result.ys:
                    continue
                ys = result.ys[y_key]
                pu.plot_xy(ax, result.x, ys, label=result.label)
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            plots_done += 1
            if legend_handles is None:
                legend_handles = ax.get_legend_handles_labels()
    fig.legend(*legend_handles, loc='lower center', ncol=NUM_COLS)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=str, nargs="*", default=None,
                        help="List of file paths to results (default=None)")
    parser.add_argument("--dps", type=str, nargs="*", default=None,
                        help="List of Parent dir of dirs containing results")
    parser.add_argument("--x_key", type=str, default="episode",
                        help=("Key to plot on X Axis and aggregate data over "
                              "(if plotting multiple files) "
                              "(default='episode')"))
    parser.add_argument("--y_keys", type=str, nargs="*",
                        default="episode_return",
                        help=("Keys to plot on Y Axis, will produce one plot "
                              "for each y-key. If 'all' will plot all keys"
                              " (default='episode_return')"))
    parser.add_argument("--average", action="store_true",
                        help=("Average over related results. This means"
                              " averaging over all listed file path when"
                              " using --fps, or over results in the same"
                              " dir when using --dps."))
    parser.add_argument("--resample", type=int, default=512,
                        help=("Number of sample points when averaging "
                              "(default=512)"))
    args = parser.parse_args()

    grouped_result_paths = []
    if args.fps is not None:
        args.fps = args.fps if isinstance(args.fps, list) else [args.fps]
        grouped_result_paths.append(args.fps)
    elif args.dps is not None:
        args.dps = args.dps if isinstance(args.dps, list) else [args.dps]
        for dp in args.dps:
            _, results_fps = list_files(dp)
            grouped_result_paths.append(results_fps)
    else:
        raise ValueError("Must supply either file paths or dir path")

    if not isinstance(args.y_keys, list):
        args.y_keys = [args.y_keys]

    all_results = load_results(
        grouped_result_paths, args.x_key, args.y_keys, args.average,
        args.resample
    )

    plot_results(all_results, args.x_key, args.y_keys)
