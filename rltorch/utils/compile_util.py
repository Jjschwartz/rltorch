"""Functions for compiling results into a single file

Result files compiled:
- solver_results.txt
- sim_results.txt
"""
import os
import os.path as osp

import rltorch.utils.file_utils as futils
from rltorch.utils.rl_logger import RESULTS_FILE, CONFIG_FILE


def compile_results(parent_dir):
    """Compiles result files in directory and sub directories """
    config_files, results_files = list_files(parent_dir)
    results_out_file = osp.join(parent_dir, RESULTS_FILE)
    compile_result_files(config_files, results_files, results_out_file)


def list_files(dir):
    config_files = []
    results_files = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            fname = osp.join(root, name)
            if fname.endswith(CONFIG_FILE):
                config_files.append(fname)
            if fname.endswith(RESULTS_FILE):
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


def compile_result_files(config_files, result_files, out_file_name):
    print(f"Compiling results into {out_file_name}")
    with open(out_file_name, "w") as fout:
        header_added = False
        for result_file in result_files:
            with open(result_file, "r") as rin:
                print(f"Reading {result_file}")
                first_line = rin.readline()
                if not header_added:
                    if is_solver_results:
                        first_line = f"Model\t{first_line}"
                    fout.write(first_line)
                    header_added = True

                line = rin.readline()
                while line:
                    if is_solver_results:
                        model_name = futils.get_dir_name(result_file)
                        line = f"{model_name}\t{line}"
                    else:
                        line = clean_sim_line(line)
                    fout.write(line)
                    line = rin.readline()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir", type=str,
                        help=("path to directory containing all directories "
                              "containing results to compile"))
    args = parser.parse_args()

    compile_results(args.parent_dir)
