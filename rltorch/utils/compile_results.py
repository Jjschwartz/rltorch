import os
import shutil

import rltorch.utils.file_utils as fu
from rltorch.utils.rl_logger import RESULTS_FILE_NAME, CONFIG_FILE_NAME, \
    RESULTS_FILE_EXT
from rltorch.user_config import DEFAULT_DATA_DIR


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_paths", type=str, nargs="*")
    parser.add_argument("--exp_name", type=str,
                        help="Experiment name (used for dir name)")
    args = parser.parse_args()

    result_fps = []
    config = None
    for d in args.dir_paths:
        d_files = fu.get_all_files_from_dir(d)
        print(d_files)
        for f in d_files:
            if config is None and \
               fu.get_file_name(f) == CONFIG_FILE_NAME:
                config = f
            if fu.get_file_name(f) == RESULTS_FILE_NAME:
                result_fps.append(f)

    new_dir = os.path.join(DEFAULT_DATA_DIR, args.exp_name)
    fu.make_dir(new_dir)

    shutil.copy(config, new_dir)
    for i, fp in enumerate(result_fps):
        fname = fu.get_dir_name(fp)
        new_fname = f"{RESULTS_FILE_NAME}_{fname}.{RESULTS_FILE_EXT}"
        new_fp = os.path.join(new_dir, new_fname)
        shutil.copy(fp, new_fp)
