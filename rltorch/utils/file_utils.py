"""This module contains general file handling functions, common to
multiple packages.
"""
import os
import yaml
import shutil
import pathlib
import tempfile
import os.path as osp


def load_yaml(file_path):
    """Load yaml file located at file path, throws error if theres an issue
    loading file.
    """
    with open(file_path) as fin:
        content = yaml.load(fin, Loader=yaml.FullLoader)
    return content


def write_yaml(file_path, data):
    """Write a dictionary to yaml file """
    with open(file_path, "w") as fout:
        yaml.dump(data, fout)


def get_file_name(file_path):
    """Extracts the file or directory name from a file path, removing
    extension.
    """
    full_file_name = file_path.split(os.sep)[-1]
    file_name = full_file_name.split(".")[0]
    return file_name


def get_dir_name(path):
    """Get the name of top level directory in path. """
    if osp.isfile(path):
        return osp.basename(osp.dirname(path))
    return osp.basename(osp.abspath(path))


def get_parent_dir_path(path):
    """Returns the path to the parent directory in path. """
    if osp.isfile(path):
        return osp.dirname(path)
    return osp.dirname(osp.abspath(path))


def make_dir(dir_path, msg=None):
    """Makes a new directory at given path, or prints warning if one already
    exists.
    """
    if osp.exists(dir_path):
        if msg is None:
            print(f"WARNING: dir {dir_path} already exists.")
        else:
            print(f"WARNING: {msg}")
    pathlib.Path(dir_path).mkdir(exist_ok=True)


def setup_sub_dir(parent_dir, sub_dir):
    """Setup save directory for single episode trace """
    new_dir = osp.join(parent_dir, sub_dir)
    make_dir(new_dir, f"sub dir already exists, storing there any way")
    return new_dir


def generate_file_path(parent_dir, file_name, extension):
    """Generates a full file path from a parent directory, file name and
    file extension.
    """
    if extension[0] != ".":
        extension = "." + extension
    return osp.join(parent_dir, file_name + extension)


def replace_extension(file_path, extension):
    """Generate a new file path from existing file path and new extension """
    split_path = file_path.rsplit(".", 1)
    return ".".join([split_path[0], extension])


def get_all_files_from_dir(dir, extension=None):
    """Returns full file paths of all files in directory with optional
    filtering by file extension.
    """
    file_list = os.listdir(dir)
    files = []
    for file_name in file_list:
        if extension is None or extension in file_name:
            files.append(osp.join(dir, file_name))
    return files


def get_tmp_file(suffix=None):
    """Returns full path to a temporary file.
    Suffix is optional file ending (e.g. .txt). Must include '.' if required.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
        return temp_file.name


def move_dirs_into_parent_dir(dir_paths, parent_dir):
    make_dir(parent_dir)
    for d in dir_paths:
        shutil.move(d, parent_dir)
