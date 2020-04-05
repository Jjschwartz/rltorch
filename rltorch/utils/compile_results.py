import os
import shutil

import rltorch.utils.file_utils as fu
import rltorch.utils.rl_logger as rllog
from rltorch.user_config import DEFAULT_DATA_DIR


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_paths", type=str, nargs="*")
    parser.add_argument("--exp_name", type=str,
                        help="Experiment name (used for dir name)")
    args = parser.parse_args()

    cfg_file = f"{rllog.CONFIG_FILE_NAME}.{rllog.CONFIG_FILE_EXT}"
    results_file = f"{rllog.RESULTS_FILE_NAME}.{rllog.RESULTS_FILE_EXT}"

    result_fps = []
    config = None
    for d in args.dir_paths:
        d_files = fu.get_all_files_from_dir(d)
        print(d_files)
        for f in d_files:
            if config is None and \
               fu.get_file_name(f) == rllog.CONFIG_FILE_NAME:
                config = f
            if fu.get_file_name(f) == rllog.RESULTS_FILE_NAME:
                result_fps.append(f)

    new_dir = os.path.join(DEFAULT_DATA_DIR, args.exp_name)
    fu.make_dir(new_dir)

    shutil.copy(config, new_dir)
    for i, fp in enumerate(result_fps):
        new_fname = f"{rllog.RESULTS_FILE_NAME}{i}.{rllog.RESULTS_FILE_EXT}"
        new_fp = os.path.join(new_dir, new_fname)
        shutil.copy(fp, new_fp)
