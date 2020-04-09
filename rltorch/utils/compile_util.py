"""Functions for compiling results into a single file

Result files compiled:
- solver_results.txt
- sim_results.txt
"""
import os
import shutil
import os.path as osp

import rltorch.utils.rl_logger as rlog
import rltorch.utils.file_utils as futils
from rltorch.user_config import DEFAULT_DATA_DIR


def compile_results(parent_dir):
    """Compiles result files in directory and sub directories """
    config_files, results_files = list_files(parent_dir)
    results_out_file = osp.join(parent_dir, rlog.RESULTS_FILE)
    compile_result_files(config_files, results_files, results_out_file)


def list_files(dir):
    config_files = []
    results_files = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            fname = osp.join(root, name)
            if fname.endswith(rlog.CONFIG_FILE):
                config_files.append(fname)
            if fname.endswith(rlog.RESULTS_FILE):
                results_files.append(fname)
    return config_files, results_files


def clean_result_line(line):
    tokens = line.split("\t")
    clean_tokens = []
    for token in tokens:
        if osp.sep in token:
            clean_tokens.append(futils.get_file_name(token))
        else:
            clean_tokens.append(token)
    return "\t".join(clean_tokens)


def format_config(config_file, headers=None):
    config = futils.load_yaml(config_file)
    tokens = []
    if headers is None:
        headers = list(config.keys())
    for h in headers:
        v = config[h]
        tokens.append(format_value(v))
    line = "\t".join(tokens)
    return headers, line


def format_value(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def compile_result_files(config_files, result_files, out_file_name):
    print(f"Compiling results into {out_file_name}")
    with open(out_file_name, "w") as fout:
        header_added = False
        config_headers = None
        for config_file, result_file in zip(config_files, result_files):
            config_headers, config_line = format_config(config_file)
            with open(result_file, "r") as rin:
                print(f"Reading {result_file}")
                result_line = rin.readline()
                if not header_added:
                    config_header_line = "\t".join(config_headers)
                    header_line = f"{config_header_line}\t{result_line}"
                    fout.write(header_line)
                    header_added = True

                result_line = rin.readline()
                while result_line:
                    result_line = clean_result_line(result_line)
                    line = f"{config_line}\t{result_line}"
                    fout.write(line)
                    result_line = rin.readline()


def copy_files_into_single_dir(dir_paths, parent_dir_name):
    result_fps = []
    config = None
    for d in dir_paths:
        d_files = futils.get_all_files_from_dir(d)
        print(d_files)
        for f in d_files:
            if config is None and \
               futils.get_file_name(f) == rlog.CONFIG_FILE_NAME:
                config = f
            if futils.get_file_name(f) == rlog.RESULTS_FILE_NAME:
                result_fps.append(f)

    new_dir = os.path.join(DEFAULT_DATA_DIR, parent_dir_name)
    futils.make_dir(new_dir)

    shutil.copy(config, new_dir)
    for i, fp in enumerate(result_fps):
        fname = futils.get_dir_name(fp)
        new_fname = f"{rlog.RESULTS_FILE_NAME}_{fname}.{rlog.RESULTS_FILE_EXT}"
        new_fp = os.path.join(new_dir, new_fname)
        shutil.copy(fp, new_fp)


def move_dirs_into_single_dir(dir_paths, parent_dir_name):
    parent_dir = os.path.join(DEFAULT_DATA_DIR, parent_dir_name)
    futils.move_dirs_into_parent_dir(dir_paths, parent_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir", type=str,
                        help=("path to directory containing all directories "
                              "containing results to compile"))
    args = parser.parse_args()

    compile_results(args.parent_dir)
